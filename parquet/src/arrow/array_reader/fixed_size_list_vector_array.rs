// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray, new_empty_array};
use arrow_schema::{DataType as ArrowType, FieldRef};

use crate::arrow::array_reader::ArrayReader;
use crate::errors::{ParquetError, Result};

/// Reader for the canonical Parquet `VECTOR` encoding.
///
/// A `VECTOR` column stores a fixed number `N` of element values per row under a
/// VECTOR-repeated middle group that contributes no definition or repetition
/// levels (`num_values == num_rows * N`). This reader wraps the leaf's primitive
/// [`ArrayReader`] and reshapes its flat output into a [`FixedSizeListArray`]:
/// requesting `R` rows reads `R * N` flat element values from the inner reader,
/// and the resulting length-`R * N` array is grouped into `R` rows of size `N`.
///
/// The wrapped column is non-nullable (vector and elements), so this reader
/// produces no definition or repetition levels.
pub struct FixedSizeListVectorArrayReader {
    /// Reader for the flattened element values
    item_reader: Box<dyn ArrayReader>,
    /// Number of element values per row (`vector_length`)
    fixed_size: usize,
    /// The `FixedSizeList` data type produced by this reader
    data_type: ArrowType,
    /// Child element field, extracted from `data_type`
    field: FieldRef,
}

impl FixedSizeListVectorArrayReader {
    /// Construct a `VECTOR` reshape reader.
    ///
    /// `data_type` must be a [`ArrowType::FixedSizeList`] whose size equals
    /// `fixed_size`.
    pub fn new(
        item_reader: Box<dyn ArrayReader>,
        fixed_size: usize,
        data_type: ArrowType,
    ) -> Result<Self> {
        if fixed_size == 0 {
            return Err(general_err!("VECTOR vector_length must be positive"));
        }
        let field = match &data_type {
            ArrowType::FixedSizeList(field, size) if *size as usize == fixed_size => field.clone(),
            other => {
                return Err(general_err!(
                    "FixedSizeListVectorArrayReader requires a FixedSizeList of size {}, got {}",
                    fixed_size,
                    other
                ));
            }
        };
        Ok(Self {
            item_reader,
            fixed_size,
            data_type,
            field,
        })
    }

    /// `rows * fixed_size`, erroring on overflow rather than wrapping.
    fn values_for(&self, rows: usize) -> Result<usize> {
        rows.checked_mul(self.fixed_size).ok_or_else(|| {
            general_err!(
                "VECTOR row count {rows} * vector_length {} overflows usize",
                self.fixed_size
            )
        })
    }
}

impl ArrayReader for FixedSizeListVectorArrayReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        // Each row maps to `fixed_size` contiguous leaf values. Because the leaf
        // has no repetition levels, the inner reader treats one value as one
        // record, and whole-vector page boundaries guarantee the returned count
        // is a multiple of `fixed_size`.
        let values_read = self
            .item_reader
            .read_records(self.values_for(batch_size)?)?;
        if values_read % self.fixed_size != 0 {
            return Err(general_err!(
                "VECTOR column read {values_read} values which is not a multiple of vector_length {}",
                self.fixed_size
            ));
        }
        Ok(values_read / self.fixed_size)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        let values = self.item_reader.consume_batch()?;
        if values.is_empty() {
            return Ok(new_empty_array(&self.data_type));
        }
        if values.len() % self.fixed_size != 0 {
            return Err(general_err!(
                "VECTOR column produced {} values which is not a multiple of vector_length {}",
                values.len(),
                self.fixed_size
            ));
        }
        let array =
            FixedSizeListArray::try_new(self.field.clone(), self.fixed_size as i32, values, None)?;
        Ok(Arc::new(array))
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        // Use decoder-level skip for whole vectors. Decoding and discarding would
        // flush the inner buffer and drop rows already read by interleaved
        // RowSelection paths.
        let skipped_values = self
            .item_reader
            .skip_records(self.values_for(num_records)?)?;
        // A clean skip lands on a whole-vector boundary.
        if skipped_values % self.fixed_size != 0 {
            return Err(general_err!(
                "VECTOR column skipped {skipped_values} values which is not a multiple of vector_length {}",
                self.fixed_size
            ));
        }
        Ok(skipped_values / self.fixed_size)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        // VECTOR columns are non-nullable and carry no definition levels.
        None
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        // VECTOR columns carry no repetition levels by construction.
        None
    }
}
