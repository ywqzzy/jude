//! Multimodal expression kernels — Rust-native, over Arrow.
//!
//! These implement the `col("x").image.decode().image.resize(...)` expression
//! chain (see `docs/multimodal_design.md`). A decoded image is represented as a
//! queryable Arrow struct `{height: int32, width: int32, channels: int32,
//! data: list<uint8>}` — SQL can read the dimensions, aggregate over them, and
//! feed the pixels onward — which also sidesteps the open question of whether a
//! `fixed_shape_tensor` extension type survives a DuckDB temp-table round-trip.
//!
//! Ops fold over an Arrow column (`ArrayRef -> ArrayRef`); nulls propagate (a
//! null input row is a null output row, never a decode of garbage). Image
//! decode/resize/encode run under the GIL-free Rust `image` crate.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, Int32Array, Int32Builder, LargeBinaryArray, ListArray,
    ListBuilder, StructArray, UInt8Builder,
};
use arrow::datatypes::{DataType, Field, Fields};

use crate::error::Error;

/// A single multimodal op in a fused chain.
#[derive(Clone, Debug)]
pub enum MmOp {
    /// Read bytes from a column of local file paths / `file://` URLs.
    UrlDownload,
    /// Decode encoded image bytes (PNG/JPEG/…) into an RGB image struct.
    ImageDecode,
    /// Resize a decoded image to (width, height).
    ImageResize { width: u32, height: u32 },
    /// Crop a decoded image to a (x, y, width, height) region.
    ImageCrop {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// Convert a decoded image's color mode ("RGB" | "L"/grayscale | "RGBA").
    ImageToMode { mode: String },
    /// No-op normalization: the struct already is the tensor representation.
    ImageToTensor,
    /// Re-encode a decoded image struct back to bytes ("PNG" | "JPEG").
    ImageEncode { format: String },
}

impl MmOp {
    /// Parse an op from its `(name, kwargs)` Python spec.
    pub fn from_spec(
        name: &str,
        get_u32: impl Fn(&str) -> Option<u32>,
        get_str: impl Fn(&str) -> Option<String>,
    ) -> Result<MmOp, Error> {
        match name {
            "url_download" => Ok(MmOp::UrlDownload),
            "image_decode" => Ok(MmOp::ImageDecode),
            "image_to_tensor" => Ok(MmOp::ImageToTensor),
            "image_resize" => {
                let width = get_u32("width")
                    .ok_or_else(|| Error::Other("image.resize requires width".into()))?;
                let height = get_u32("height")
                    .ok_or_else(|| Error::Other("image.resize requires height".into()))?;
                Ok(MmOp::ImageResize { width, height })
            }
            "image_crop" => {
                let x = get_u32("x").ok_or_else(|| Error::Other("image.crop requires x".into()))?;
                let y = get_u32("y").ok_or_else(|| Error::Other("image.crop requires y".into()))?;
                let width = get_u32("width")
                    .ok_or_else(|| Error::Other("image.crop requires width".into()))?;
                let height = get_u32("height")
                    .ok_or_else(|| Error::Other("image.crop requires height".into()))?;
                Ok(MmOp::ImageCrop {
                    x,
                    y,
                    width,
                    height,
                })
            }
            "image_to_mode" => {
                let mode = get_str("mode")
                    .ok_or_else(|| Error::Other("image.to_mode requires mode".into()))?;
                Ok(MmOp::ImageToMode { mode })
            }
            "image_encode" => {
                let format = get_str("format").unwrap_or_else(|| "PNG".to_string());
                Ok(MmOp::ImageEncode { format })
            }
            other => Err(Error::Other(format!("unknown multimodal op: {other}"))),
        }
    }
}

/// The Arrow struct type of a decoded image.
fn image_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("height", DataType::Int32, true),
        Field::new("width", DataType::Int32, true),
        Field::new("channels", DataType::Int32, true),
        Field::new(
            "data",
            DataType::List(Arc::new(Field::new("item", DataType::UInt8, true))),
            true,
        ),
    ])
}

/// Fold an op chain over a single input column, returning the output column.
pub fn apply_chain(input: &ArrayRef, ops: &[MmOp]) -> Result<ArrayRef, Error> {
    let mut col = input.clone();
    for op in ops {
        col = match op {
            MmOp::UrlDownload => url_download(&col)?,
            MmOp::ImageDecode => image_decode(&col)?,
            MmOp::ImageToTensor => col, // struct already is the tensor form
            MmOp::ImageResize { width, height } => image_resize(&col, *width, *height)?,
            MmOp::ImageCrop {
                x,
                y,
                width,
                height,
            } => image_crop(&col, *x, *y, *width, *height)?,
            MmOp::ImageToMode { mode } => image_to_mode(&col, mode)?,
            MmOp::ImageEncode { format } => image_encode(&col, format)?,
        };
    }
    Ok(col)
}

/// Read bytes from a column of local file paths / `file://` URLs into a binary
/// column. Remote schemes (`http(s)://`, `s3://`, `gs://`) are not yet
/// supported here — they return an explicit error rather than silently failing.
fn url_download(input: &ArrayRef) -> Result<ArrayRef, Error> {
    let paths = string_rows(input)?;
    let mut b = arrow::array::BinaryBuilder::new();
    for p in paths {
        match p {
            None => b.append_null(),
            Some(path) => {
                let local = path.strip_prefix("file://").unwrap_or(&path);
                if local.contains("://") {
                    return Err(Error::Other(format!(
                        "url.download: remote scheme not yet supported: {path}"
                    )));
                }
                let bytes = std::fs::read(local)
                    .map_err(|e| Error::Other(format!("url.download failed for {local}: {e}")))?;
                b.append_value(&bytes);
            }
        }
    }
    Ok(Arc::new(b.finish()))
}

/// Read a string/large-string column into optional path rows.
fn string_rows(input: &ArrayRef) -> Result<Vec<Option<String>>, Error> {
    use arrow::array::{LargeStringArray, StringArray};
    if let Some(a) = input.as_any().downcast_ref::<StringArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .collect());
    }
    if let Some(a) = input.as_any().downcast_ref::<LargeStringArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .collect());
    }
    Err(Error::Other(
        "url.download expects a string column of paths/URLs".into(),
    ))
}

/// Coerce a binary-ish array into an accessor over its byte rows.
fn as_binary_rows(input: &ArrayRef) -> Result<Vec<Option<Vec<u8>>>, Error> {
    if let Some(a) = input.as_any().downcast_ref::<BinaryArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_vec())
                }
            })
            .collect());
    }
    if let Some(a) = input.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok((0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_vec())
                }
            })
            .collect());
    }
    Err(Error::Other(
        "image.decode expects a binary column of encoded image bytes".into(),
    ))
}

/// Decode encoded image bytes → struct{height, width, channels, data}.
fn image_decode(input: &ArrayRef) -> Result<ArrayRef, Error> {
    let rows = as_binary_rows(input)?;
    build_image_struct(rows.into_iter().map(|maybe| {
        maybe
            .map(|bytes| {
                let img = image::load_from_memory(&bytes)
                    .map_err(|e| Error::Other(format!("image decode failed: {e}")))?
                    .to_rgb8();
                let (w, h) = (img.width(), img.height());
                Ok::<_, Error>(DecodedImage {
                    width: w,
                    height: h,
                    channels: 3,
                    pixels: img.into_raw(),
                })
            })
            .transpose()
    }))
}

/// Resize each decoded image to (width, height).
fn image_resize(input: &ArrayRef, width: u32, height: u32) -> Result<ArrayRef, Error> {
    let imgs = read_image_struct(input)?;
    build_image_struct(imgs.into_iter().map(|maybe| {
        maybe
            .map(|d| {
                let buf = image::RgbImage::from_raw(d.width, d.height, d.pixels)
                    .ok_or_else(|| Error::Other("corrupt decoded image buffer".into()))?;
                let resized = image::imageops::resize(
                    &buf,
                    width,
                    height,
                    image::imageops::FilterType::Triangle,
                );
                Ok::<_, Error>(DecodedImage {
                    width,
                    height,
                    channels: 3,
                    pixels: resized.into_raw(),
                })
            })
            .transpose()
    }))
}

/// Reconstruct a channel-correct `DynamicImage` from a decoded struct row
/// (1 → grayscale, 3 → RGB, 4 → RGBA).
fn decoded_to_dynamic(d: &DecodedImage) -> Result<image::DynamicImage, Error> {
    let err = || Error::Other("corrupt decoded image buffer".into());
    Ok(match d.channels {
        1 => image::DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(d.width, d.height, d.pixels.clone()).ok_or_else(err)?,
        ),
        4 => image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(d.width, d.height, d.pixels.clone()).ok_or_else(err)?,
        ),
        _ => image::DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(d.width, d.height, d.pixels.clone()).ok_or_else(err)?,
        ),
    })
}

/// Flatten a `DynamicImage` back into a decoded struct row (channels from its
/// color type; other modes normalized to RGB8).
fn dynamic_to_decoded(img: image::DynamicImage) -> DecodedImage {
    use image::DynamicImage::*;
    match img {
        ImageLuma8(b) => DecodedImage {
            width: b.width(),
            height: b.height(),
            channels: 1,
            pixels: b.into_raw(),
        },
        ImageRgba8(b) => DecodedImage {
            width: b.width(),
            height: b.height(),
            channels: 4,
            pixels: b.into_raw(),
        },
        ImageRgb8(b) => DecodedImage {
            width: b.width(),
            height: b.height(),
            channels: 3,
            pixels: b.into_raw(),
        },
        other => {
            let b = other.to_rgb8();
            DecodedImage {
                width: b.width(),
                height: b.height(),
                channels: 3,
                pixels: b.into_raw(),
            }
        }
    }
}

/// Crop each decoded image to (x, y, width, height).
fn image_crop(
    input: &ArrayRef,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<ArrayRef, Error> {
    let imgs = read_image_struct(input)?;
    build_image_struct(imgs.into_iter().map(|maybe| {
        maybe
            .map(|d| {
                let img = decoded_to_dynamic(&d)?;
                // crop_imm preserves the color mode (channels).
                let cropped = img.crop_imm(x, y, width, height);
                Ok::<_, Error>(dynamic_to_decoded(cropped))
            })
            .transpose()
    }))
}

/// Convert each decoded image's color mode ("RGB" | "L"/grayscale | "RGBA").
fn image_to_mode(input: &ArrayRef, mode: &str) -> Result<ArrayRef, Error> {
    let imgs = read_image_struct(input)?;
    build_image_struct(imgs.into_iter().map(|maybe| {
        maybe
            .map(|d| {
                let img = decoded_to_dynamic(&d)?;
                let converted = match mode.to_ascii_uppercase().as_str() {
                    "L" | "GRAY" | "GREY" | "GRAYSCALE" => {
                        image::DynamicImage::ImageLuma8(img.to_luma8())
                    }
                    "RGBA" => image::DynamicImage::ImageRgba8(img.to_rgba8()),
                    "RGB" => image::DynamicImage::ImageRgb8(img.to_rgb8()),
                    other => return Err(Error::Other(format!("unsupported image mode: {other}"))),
                };
                Ok::<_, Error>(dynamic_to_decoded(converted))
            })
            .transpose()
    }))
}

/// Re-encode each decoded image struct to bytes.
fn image_encode(input: &ArrayRef, format: &str) -> Result<ArrayRef, Error> {
    let imgs = read_image_struct(input)?;
    let fmt = match format.to_ascii_uppercase().as_str() {
        "PNG" => image::ImageFormat::Png,
        "JPEG" | "JPG" => image::ImageFormat::Jpeg,
        "BMP" => image::ImageFormat::Bmp,
        other => {
            return Err(Error::Other(format!(
                "unsupported image encode format: {other}"
            )))
        }
    };
    let mut builder = arrow::array::BinaryBuilder::new();
    for maybe in imgs {
        match maybe {
            None => builder.append_null(),
            Some(d) => {
                let img = decoded_to_dynamic(&d)?;
                let mut out = std::io::Cursor::new(Vec::new());
                img.write_to(&mut out, fmt)
                    .map_err(|e| Error::Other(format!("image encode failed: {e}")))?;
                builder.append_value(out.into_inner());
            }
        }
    }
    Ok(Arc::new(builder.finish()))
}

struct DecodedImage {
    width: u32,
    height: u32,
    channels: i32,
    pixels: Vec<u8>,
}

/// Assemble a struct array from an iterator of optional decoded images.
fn build_image_struct(
    items: impl Iterator<Item = Result<Option<DecodedImage>, Error>>,
) -> Result<ArrayRef, Error> {
    let mut h = Int32Builder::new();
    let mut w = Int32Builder::new();
    let mut c = Int32Builder::new();
    let mut data = ListBuilder::new(UInt8Builder::new());
    let mut validity: Vec<bool> = Vec::new();
    for item in items {
        match item? {
            None => {
                h.append_null();
                w.append_null();
                c.append_null();
                data.append_null();
                validity.push(false);
            }
            Some(d) => {
                h.append_value(d.height as i32);
                w.append_value(d.width as i32);
                c.append_value(d.channels);
                data.values().append_slice(&d.pixels);
                data.append(true);
                validity.push(true);
            }
        }
    }
    let fields = image_struct_fields();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(h.finish()),
        Arc::new(w.finish()),
        Arc::new(c.finish()),
        Arc::new(data.finish()),
    ];
    let null_buffer = arrow::buffer::NullBuffer::from(validity);
    let struct_arr = StructArray::new(fields, arrays, Some(null_buffer));
    Ok(Arc::new(struct_arr))
}

/// Read a decoded-image struct array back into DecodedImage rows.
fn read_image_struct(input: &ArrayRef) -> Result<Vec<Option<DecodedImage>>, Error> {
    let st = input
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            Error::Other("expected a decoded-image struct (call .image.decode() first)".into())
        })?;
    let h = st
        .column_by_name("height")
        .and_then(|a| a.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| Error::Other("image struct missing height".into()))?;
    let w = st
        .column_by_name("width")
        .and_then(|a| a.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| Error::Other("image struct missing width".into()))?;
    let c = st
        .column_by_name("channels")
        .and_then(|a| a.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| Error::Other("image struct missing channels".into()))?;
    let data = st
        .column_by_name("data")
        .and_then(|a| a.as_any().downcast_ref::<ListArray>())
        .ok_or_else(|| Error::Other("image struct missing data".into()))?;
    let mut out = Vec::with_capacity(st.len());
    for i in 0..st.len() {
        if st.is_null(i) {
            out.push(None);
            continue;
        }
        let pixels_arr = data.value(i);
        let pixels = pixels_arr
            .as_any()
            .downcast_ref::<arrow::array::UInt8Array>()
            .ok_or_else(|| Error::Other("image data is not uint8".into()))?;
        let pixels: Vec<u8> = pixels.values().to_vec();
        out.push(Some(DecodedImage {
            width: w.value(i) as u32,
            height: h.value(i) as u32,
            channels: c.value(i),
            pixels,
        }));
    }
    Ok(out)
}

/// A tiny 2x2 RGB PNG, generated in-memory (no fixture file).
#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny 2x2 RGB PNG, generated in-memory (no fixture file).
    fn tiny_png() -> Vec<u8> {
        let mut img = image::RgbImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        img.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        img.put_pixel(1, 1, image::Rgb([255, 255, 255]));
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    fn binary_col(rows: Vec<Option<Vec<u8>>>) -> ArrayRef {
        let mut b = arrow::array::BinaryBuilder::new();
        for r in rows {
            match r {
                None => b.append_null(),
                Some(v) => b.append_value(&v),
            }
        }
        Arc::new(b.finish())
    }

    #[test]
    fn decode_reports_dimensions() {
        let col = binary_col(vec![Some(tiny_png()), None]);
        let out = apply_chain(&col, &[MmOp::ImageDecode]).unwrap();
        let imgs = read_image_struct(&out).unwrap();
        assert_eq!(imgs.len(), 2);
        let d = imgs[0].as_ref().unwrap();
        assert_eq!((d.width, d.height, d.channels), (2, 2, 3));
        assert_eq!(d.pixels.len(), 2 * 2 * 3);
        // first pixel is red
        assert_eq!(&d.pixels[0..3], &[255, 0, 0]);
        // null propagates
        assert!(imgs[1].is_none());
    }

    #[test]
    fn resize_changes_dimensions() {
        let col = binary_col(vec![Some(tiny_png())]);
        let out = apply_chain(
            &col,
            &[
                MmOp::ImageDecode,
                MmOp::ImageResize {
                    width: 4,
                    height: 4,
                },
            ],
        )
        .unwrap();
        let d = read_image_struct(&out).unwrap()[0]
            .as_ref()
            .unwrap()
            .pixels
            .len();
        assert_eq!(d, 4 * 4 * 3);
    }

    #[test]
    fn encode_round_trips() {
        let col = binary_col(vec![Some(tiny_png())]);
        let out = apply_chain(
            &col,
            &[
                MmOp::ImageDecode,
                MmOp::ImageEncode {
                    format: "PNG".into(),
                },
            ],
        )
        .unwrap();
        let bytes = as_binary_rows(&out).unwrap();
        // re-decode the encoded bytes; dimensions must survive
        let redec = apply_chain(&binary_col(bytes), &[MmOp::ImageDecode]).unwrap();
        let imgs = read_image_struct(&redec).unwrap();
        let d = imgs[0].as_ref().unwrap();
        assert_eq!((d.width, d.height), (2, 2));
    }

    #[test]
    fn crop_changes_dimensions() {
        let col = binary_col(vec![Some(tiny_png())]);
        let out = apply_chain(
            &col,
            &[
                MmOp::ImageDecode,
                MmOp::ImageCrop {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                },
            ],
        )
        .unwrap();
        let imgs = read_image_struct(&out).unwrap();
        let d = imgs[0].as_ref().unwrap();
        assert_eq!((d.width, d.height, d.channels), (1, 1, 3));
        assert_eq!(d.pixels.len(), 3);
        assert_eq!(&d.pixels[0..3], &[255, 0, 0]); // top-left is red
    }

    #[test]
    fn to_mode_grayscale_is_one_channel() {
        let col = binary_col(vec![Some(tiny_png())]);
        let out = apply_chain(
            &col,
            &[MmOp::ImageDecode, MmOp::ImageToMode { mode: "L".into() }],
        )
        .unwrap();
        let imgs = read_image_struct(&out).unwrap();
        let d = imgs[0].as_ref().unwrap();
        assert_eq!((d.width, d.height, d.channels), (2, 2, 1));
        assert_eq!(d.pixels.len(), 2 * 2 * 1);
    }

    #[test]
    fn to_mode_rgba_is_four_channels_and_encodes() {
        let col = binary_col(vec![Some(tiny_png())]);
        // decode -> RGBA -> encode PNG -> re-decode; RGBA encode must not error.
        let out = apply_chain(
            &col,
            &[
                MmOp::ImageDecode,
                MmOp::ImageToMode {
                    mode: "RGBA".into(),
                },
            ],
        )
        .unwrap();
        let d0 = read_image_struct(&out).unwrap();
        assert_eq!(d0[0].as_ref().unwrap().channels, 4);
        let enc = apply_chain(
            &out,
            &[MmOp::ImageEncode {
                format: "PNG".into(),
            }],
        )
        .unwrap();
        assert_eq!(as_binary_rows(&enc).unwrap().len(), 1);
    }
}
