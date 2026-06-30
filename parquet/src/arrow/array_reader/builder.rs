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

use std::sync::{Arc, RwLock};

use arrow_schema::{DataType, Fields, SchemaBuilder};

use crate::arrow::ProjectionMask;
use crate::arrow::array_reader::byte_view_array::make_byte_view_array_reader;
use crate::arrow::array_reader::cached_array_reader::CacheRole;
use crate::arrow::array_reader::cached_array_reader::CachedArrayReader;
use crate::arrow::array_reader::empty_array::make_empty_array_reader;
use crate::arrow::array_reader::fixed_len_byte_array::make_fixed_len_byte_array_reader;
use crate::arrow::array_reader::row_group_cache::RowGroupCache;
use crate::arrow::array_reader::row_group_index::RowGroupIndexReader;
use crate::arrow::array_reader::row_number::RowNumberReader;
use crate::arrow::array_reader::{
    ArrayReader, FixedSizeListArrayReader, FixedSizeListVectorArrayReader, ListArrayReader,
    ListViewArrayReader, MapArrayReader, NullArrayReader, PrimitiveArrayReader, RowGroups,
    StructArrayReader, make_byte_array_dictionary_reader, make_byte_array_reader,
};
use crate::arrow::arrow_reader::DEFAULT_BATCH_SIZE;
use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use crate::arrow::schema::{ParquetField, ParquetFieldType, VirtualColumnType};
use crate::basic::Type as PhysicalType;
use crate::column::page::{Page, PageIterator, PageMetadata, PageReader};
use crate::data_type::{BoolType, DoubleType, FloatType, Int32Type, Int64Type, Int96Type};
use crate::errors::{ParquetError, Result};
use crate::file::metadata::{ParquetMetaData, RowGroupMetaData};
use crate::schema::types::{ColumnDescriptor, ColumnPath, Type};

/// Builder for [`CacheOptions`]
#[derive(Debug, Clone)]
pub struct CacheOptionsBuilder<'a> {
    /// Projection mask to apply to the cache
    pub projection_mask: &'a ProjectionMask,
    /// Cache to use for storing row groups
    pub cache: &'a Arc<RwLock<RowGroupCache>>,
}

impl<'a> CacheOptionsBuilder<'a> {
    /// create a new cache options builder
    pub fn new(projection_mask: &'a ProjectionMask, cache: &'a Arc<RwLock<RowGroupCache>>) -> Self {
        Self {
            projection_mask,
            cache,
        }
    }

    /// Return a new [`CacheOptions`] for producing (populating) the cache
    pub fn producer(self) -> CacheOptions<'a> {
        CacheOptions {
            projection_mask: self.projection_mask,
            cache: self.cache,
            role: CacheRole::Producer,
        }
    }

    /// return a new [`CacheOptions`] for consuming (reading) the cache
    pub fn consumer(self) -> CacheOptions<'a> {
        CacheOptions {
            projection_mask: self.projection_mask,
            cache: self.cache,
            role: CacheRole::Consumer,
        }
    }
}

/// Cache options containing projection mask, cache, and role
#[derive(Clone)]
pub struct CacheOptions<'a> {
    pub projection_mask: &'a ProjectionMask,
    pub cache: &'a Arc<RwLock<RowGroupCache>>,
    pub role: CacheRole,
}

/// Builds [`ArrayReader`]s from parquet schema, projection mask, and RowGroups reader
pub struct ArrayReaderBuilder<'a> {
    /// Source of row group data
    row_groups: &'a dyn RowGroups,
    /// Optional cache options for the array reader
    cache_options: Option<&'a CacheOptions<'a>>,
    /// Parquet metadata for computing virtual column values
    parquet_metadata: Option<&'a ParquetMetaData>,
    /// metrics
    metrics: &'a ArrowReaderMetrics,
    /// Batch size for pre-allocating internal buffers
    batch_size: usize,
}

/// Arguments threaded through the recursive `build_*_reader` calls.
///
/// Bundling the per-field arguments into a single struct keeps the recursive
/// builder signatures small and provides one documented place to add new
/// per-reader options in the future.
#[derive(Clone, Copy)]
struct ReaderArgs<'a> {
    /// The parquet field the output array corresponds to.
    field: &'a ParquetField,
    /// Which leaf columns are being read.
    mask: &'a ProjectionMask,
}

impl<'a> ReaderArgs<'a> {
    /// Returns a copy of these arguments pointing at `field`.
    ///
    /// Used when recursing from a field into one of its children.
    fn with_field(self, field: &'a ParquetField) -> Self {
        Self { field, ..self }
    }
}

fn validate_vector_column_chunks<'a>(
    row_groups: impl IntoIterator<Item = &'a RowGroupMetaData>,
    col_idx: usize,
    vector_length: i32,
) -> Result<()> {
    if vector_length <= 0 {
        return Err(general_err!(
            "VECTOR column {} has invalid vector_length {}",
            col_idx,
            vector_length
        ));
    }
    let vector_length = i64::from(vector_length);

    for (row_group_idx, row_group) in row_groups.into_iter().enumerate() {
        let rows = row_group.num_rows();
        if rows < 0 {
            return Err(general_err!(
                "VECTOR column {} row group {} has negative num_rows {}",
                col_idx,
                row_group_idx,
                rows
            ));
        }
        let expected = rows.checked_mul(vector_length).ok_or_else(|| {
            general_err!(
                "VECTOR column {} row group {} num_rows {} * vector_length {} overflows i64",
                col_idx,
                row_group_idx,
                rows,
                vector_length
            )
        })?;
        let column = row_group.columns().get(col_idx).ok_or_else(|| {
            general_err!(
                "VECTOR column {} missing from row group {} metadata",
                col_idx,
                row_group_idx
            )
        })?;
        let actual = column.num_values();
        if actual != expected {
            return Err(general_err!(
                "VECTOR column {} row group {} has num_values {}, expected {} (= num_rows {} * vector_length {})",
                col_idx,
                row_group_idx,
                actual,
                expected,
                rows,
                vector_length
            ));
        }
    }

    Ok(())
}

fn validate_vector_page_value_count(
    col_idx: usize,
    row_group_idx: usize,
    num_values: usize,
    num_rows: Option<usize>,
    vector_length: usize,
) -> Result<()> {
    if vector_length == 0 {
        return Err(general_err!(
            "VECTOR column {} has invalid vector_length 0",
            col_idx
        ));
    }

    if num_values % vector_length != 0 {
        return Err(general_err!(
            "VECTOR column {} row group {} page has num_values {}, which is not a multiple of vector_length {}",
            col_idx,
            row_group_idx,
            num_values,
            vector_length
        ));
    }

    if let Some(num_rows) = num_rows {
        let expected = num_rows.checked_mul(vector_length).ok_or_else(|| {
            general_err!(
                "VECTOR column {} row group {} page num_rows {} * vector_length {} overflows usize",
                col_idx,
                row_group_idx,
                num_rows,
                vector_length
            )
        })?;
        if num_values != expected {
            return Err(general_err!(
                "VECTOR column {} row group {} page has num_values {}, expected {} (= num_rows {} * vector_length {})",
                col_idx,
                row_group_idx,
                num_values,
                expected,
                num_rows,
                vector_length
            ));
        }
    }

    Ok(())
}

fn validate_vector_page_metadata(
    col_idx: usize,
    row_group_idx: usize,
    metadata: &PageMetadata,
    vector_length: usize,
) -> Result<()> {
    if metadata.is_dict {
        return Ok(());
    }

    if let Some(num_nulls) = metadata.num_nulls.filter(|&n| n != 0) {
        return Err(general_err!(
            "VECTOR column {} row group {} page has {} null values; VECTOR pages must be dense",
            col_idx,
            row_group_idx,
            num_nulls
        ));
    }

    if let Some(num_values) = metadata.num_levels {
        validate_vector_page_value_count(
            col_idx,
            row_group_idx,
            num_values,
            metadata.num_rows,
            vector_length,
        )?;
    }

    Ok(())
}

fn validate_vector_page(
    col_idx: usize,
    row_group_idx: usize,
    page: &Page,
    vector_length: usize,
) -> Result<()> {
    match page {
        Page::DataPage { num_values, .. } => validate_vector_page_value_count(
            col_idx,
            row_group_idx,
            *num_values as usize,
            None,
            vector_length,
        ),
        Page::DataPageV2 {
            num_values,
            num_rows,
            num_nulls,
            ..
        } => {
            if *num_nulls != 0 {
                return Err(general_err!(
                    "VECTOR column {} row group {} page has {} null values; VECTOR pages must be dense",
                    col_idx,
                    row_group_idx,
                    num_nulls
                ));
            }
            validate_vector_page_value_count(
                col_idx,
                row_group_idx,
                *num_values as usize,
                Some(*num_rows as usize),
                vector_length,
            )
        }
        Page::DictionaryPage { .. } => Ok(()),
    }
}

struct VectorPageIterator {
    inner: Box<dyn PageIterator>,
    col_idx: usize,
    vector_length: usize,
    row_group_idx: usize,
}

impl VectorPageIterator {
    fn new(inner: Box<dyn PageIterator>, col_idx: usize, vector_length: usize) -> Self {
        Self {
            inner,
            col_idx,
            vector_length,
            row_group_idx: 0,
        }
    }
}

impl Iterator for VectorPageIterator {
    type Item = Result<Box<dyn PageReader>>;

    fn next(&mut self) -> Option<Self::Item> {
        let row_group_idx = self.row_group_idx;
        self.row_group_idx += 1;
        self.inner.next().map(|reader| {
            reader.map(|inner| {
                Box::new(VectorPageReader {
                    inner,
                    col_idx: self.col_idx,
                    vector_length: self.vector_length,
                    row_group_idx,
                }) as Box<dyn PageReader>
            })
        })
    }
}

impl PageIterator for VectorPageIterator {}

struct VectorPageReader {
    inner: Box<dyn PageReader>,
    col_idx: usize,
    vector_length: usize,
    row_group_idx: usize,
}

impl Iterator for VectorPageReader {
    type Item = Result<Page>;

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_page().transpose()
    }
}

impl PageReader for VectorPageReader {
    fn get_next_page(&mut self) -> Result<Option<Page>> {
        let page = self.inner.get_next_page()?;
        if let Some(page) = page.as_ref() {
            validate_vector_page(self.col_idx, self.row_group_idx, page, self.vector_length)?;
        }
        Ok(page)
    }

    fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
        let metadata = self.inner.peek_next_page()?;
        if let Some(metadata) = metadata.as_ref() {
            validate_vector_page_metadata(
                self.col_idx,
                self.row_group_idx,
                metadata,
                self.vector_length,
            )?;
        }
        Ok(metadata)
    }

    fn skip_next_page(&mut self) -> Result<()> {
        let metadata = self.peek_next_page()?;
        if metadata
            .as_ref()
            .is_some_and(|m| !m.is_dict && m.num_levels.is_none())
        {
            // Offset-index metadata may lack page-header counts/nulls; read the
            // page so VECTOR invariants are validated before skipping it.
            let _ = self.get_next_page()?;
            return Ok(());
        }
        self.inner.skip_next_page()
    }

    fn at_record_boundary(&mut self) -> Result<bool> {
        self.peek_next_page()?;
        self.inner.at_record_boundary()
    }
}

impl<'a> ArrayReaderBuilder<'a> {
    /// Create a new `ArrayReaderBuilder`
    pub fn new(row_groups: &'a dyn RowGroups, metrics: &'a ArrowReaderMetrics) -> Self {
        Self {
            row_groups,
            cache_options: None,
            parquet_metadata: None,
            metrics,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }

    /// Set the batch size used to pre-allocate internal buffers.
    ///
    /// This avoids reallocations when reading the first batch of data.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Add cache options to the builder
    pub fn with_cache_options(mut self, cache_options: Option<&'a CacheOptions<'a>>) -> Self {
        self.cache_options = cache_options;
        self
    }

    /// Add parquet metadata to the builder for computing virtual column values
    pub fn with_parquet_metadata(mut self, parquet_metadata: &'a ParquetMetaData) -> Self {
        self.parquet_metadata = Some(parquet_metadata);
        self
    }

    /// Create [`ArrayReader`] from parquet schema, projection mask, and parquet file reader.
    pub fn build_array_reader(
        &self,
        field: Option<&ParquetField>,
        mask: &ProjectionMask,
    ) -> Result<Box<dyn ArrayReader>> {
        let reader = field
            .and_then(|field| self.build_reader(ReaderArgs { field, mask }).transpose())
            .transpose()?
            .unwrap_or_else(|| make_empty_array_reader(self.num_rows()));

        Ok(reader)
    }

    /// Return the total number of rows
    fn num_rows(&self) -> usize {
        self.row_groups.num_rows()
    }

    fn build_reader(&self, args: ReaderArgs<'_>) -> Result<Option<Box<dyn ArrayReader>>> {
        match args.field.field_type {
            ParquetFieldType::Primitive { col_idx, .. } => {
                let Some(reader) = self.build_primitive_reader(args)? else {
                    return Ok(None);
                };
                let Some(cache_options) = self.cache_options.as_ref() else {
                    return Ok(Some(reader));
                };

                if cache_options.projection_mask.leaf_included(col_idx) {
                    Ok(Some(Box::new(CachedArrayReader::new(
                        reader,
                        Arc::clone(cache_options.cache),
                        col_idx,
                        cache_options.role,
                        self.metrics.clone(), // cheap clone
                    ))))
                } else {
                    Ok(Some(reader))
                }
            }
            ParquetFieldType::Virtual(virtual_type) => {
                // Virtual columns don't have data in the parquet file
                // They need to be built by specialized readers
                match virtual_type {
                    VirtualColumnType::RowNumber => Ok(Some(self.build_row_number_reader()?)),
                    VirtualColumnType::RowGroupIndex => {
                        Ok(Some(self.build_row_group_index_reader()?))
                    }
                }
            }
            ParquetFieldType::Group { .. } => match &args.field.arrow_type {
                DataType::Map(_, _) => self.build_map_reader(args),
                DataType::Struct(_) => self.build_struct_reader(args),
                DataType::List(_)
                | DataType::LargeList(_)
                | DataType::ListView(_)
                | DataType::LargeListView(_) => self.build_list_reader(args),
                DataType::FixedSizeList(_, _) => self.build_fixed_size_list_reader(args),
                d => unimplemented!("reading group type {} not implemented", d),
            },
        }
    }

    fn build_row_number_reader(&self) -> Result<Box<dyn ArrayReader>> {
        let parquet_metadata = self.parquet_metadata.ok_or_else(|| {
            ParquetError::General(
                "ParquetMetaData is required to read virtual row number columns.".to_string(),
            )
        })?;
        Ok(Box::new(RowNumberReader::try_new(
            parquet_metadata,
            self.row_groups.row_groups(),
        )?))
    }

    fn build_row_group_index_reader(&self) -> Result<Box<dyn ArrayReader>> {
        let parquet_metadata = self.parquet_metadata.ok_or_else(|| {
            ParquetError::General(
                "ParquetMetaData is required to read virtual row group index columns.".to_string(),
            )
        })?;
        Ok(Box::new(RowGroupIndexReader::try_new(
            parquet_metadata,
            self.row_groups.row_groups(),
        )?))
    }

    /// Build array reader for map type.
    fn build_map_reader(&self, args: ReaderArgs<'_>) -> Result<Option<Box<dyn ArrayReader>>> {
        let field = args.field;
        let children = field.children().unwrap();
        assert_eq!(children.len(), 2);

        let key_reader = self.build_reader(args.with_field(&children[0]))?;
        let value_reader = self.build_reader(args.with_field(&children[1]))?;

        match (key_reader, value_reader) {
            (Some(key_reader), Some(value_reader)) => {
                // Need to retrieve underlying data type to handle projection
                let key_type = key_reader.get_data_type().clone();
                let value_type = value_reader.get_data_type().clone();

                let data_type = match &field.arrow_type {
                    DataType::Map(map_field, is_sorted) => match map_field.data_type() {
                        DataType::Struct(fields) => {
                            assert_eq!(fields.len(), 2);
                            let struct_field = map_field.as_ref().clone().with_data_type(
                                DataType::Struct(Fields::from(vec![
                                    fields[0].as_ref().clone().with_data_type(key_type),
                                    fields[1].as_ref().clone().with_data_type(value_type),
                                ])),
                            );
                            DataType::Map(Arc::new(struct_field), *is_sorted)
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };

                Ok(Some(Box::new(MapArrayReader::new(
                    key_reader,
                    value_reader,
                    data_type,
                    field.def_level,
                    field.rep_level,
                    field.nullable,
                ))))
            }
            (None, None) => Ok(None),
            _ => Err(general_err!(
                "partial projection of MapArray is not supported"
            )),
        }
    }

    /// Build array reader for list type.
    fn build_list_reader(&self, args: ReaderArgs<'_>) -> Result<Option<Box<dyn ArrayReader>>> {
        let field = args.field;
        let children = field.children().unwrap();
        assert_eq!(children.len(), 1);

        let reader = match self.build_reader(args.with_field(&children[0]))? {
            Some(item_reader) => {
                // Need to retrieve underlying data type to handle projection
                let item_type = item_reader.get_data_type().clone();
                let reader: Box<dyn ArrayReader> = match &field.arrow_type {
                    DataType::List(f) => {
                        let data_type =
                            DataType::List(Arc::new(f.as_ref().clone().with_data_type(item_type)));
                        Box::new(ListArrayReader::<i32>::new(
                            item_reader,
                            data_type,
                            field.def_level,
                            field.rep_level,
                            field.nullable,
                        ))
                    }
                    DataType::LargeList(f) => {
                        let data_type = DataType::LargeList(Arc::new(
                            f.as_ref().clone().with_data_type(item_type),
                        ));
                        Box::new(ListArrayReader::<i64>::new(
                            item_reader,
                            data_type,
                            field.def_level,
                            field.rep_level,
                            field.nullable,
                        ))
                    }
                    DataType::ListView(f) => {
                        let data_type = DataType::ListView(Arc::new(
                            f.as_ref().clone().with_data_type(item_type),
                        ));
                        Box::new(ListViewArrayReader::<i32>::new(
                            item_reader,
                            data_type,
                            field.def_level,
                            field.rep_level,
                            field.nullable,
                        ))
                    }
                    DataType::LargeListView(f) => {
                        let data_type = DataType::LargeListView(Arc::new(
                            f.as_ref().clone().with_data_type(item_type),
                        ));
                        Box::new(ListViewArrayReader::<i64>::new(
                            item_reader,
                            data_type,
                            field.def_level,
                            field.rep_level,
                            field.nullable,
                        ))
                    }
                    _ => unreachable!(),
                };
                Some(reader)
            }
            None => None,
        };
        Ok(reader)
    }

    /// Build array reader for fixed-size list type.
    fn build_fixed_size_list_reader(
        &self,
        args: ReaderArgs<'_>,
    ) -> Result<Option<Box<dyn ArrayReader>>> {
        let field = args.field;
        let children = field.children().unwrap();
        assert_eq!(children.len(), 1);

        let child = &children[0];
        let vector_length = match &child.field_type {
            ParquetFieldType::Primitive { vector_length, .. } => *vector_length,
            _ => None,
        };

        let reader = match self.build_reader(args.with_field(child))? {
            Some(item_reader) => {
                let item_type = item_reader.get_data_type().clone();
                let reader = match &field.arrow_type {
                    &DataType::FixedSizeList(ref f, size) => {
                        let data_type = DataType::FixedSizeList(
                            Arc::new(f.as_ref().clone().with_data_type(item_type)),
                            size,
                        );

                        if let Some(n) = vector_length {
                            if n != size {
                                return Err(general_err!(
                                    "VECTOR column has Parquet vector_length {} but Arrow FixedSizeList size {}",
                                    n,
                                    size
                                ));
                            }
                            if field.nullable
                                || field.def_level != 0
                                || field.rep_level != 0
                                || child.def_level != 0
                                || child.rep_level != 0
                                || f.is_nullable()
                            {
                                return Err(general_err!(
                                    "reading nullable or nested VECTOR columns is not supported"
                                ));
                            }
                            Box::new(FixedSizeListVectorArrayReader::new(
                                item_reader,
                                n as usize,
                                data_type,
                            )?) as _
                        } else {
                            Box::new(FixedSizeListArrayReader::new(
                                item_reader,
                                size as usize,
                                data_type,
                                field.def_level,
                                field.rep_level,
                                field.nullable,
                            )) as _
                        }
                    }
                    _ => unimplemented!(),
                };
                Some(reader)
            }
            None => None,
        };
        Ok(reader)
    }

    /// Creates primitive array reader for each primitive type.
    fn build_primitive_reader(&self, args: ReaderArgs<'_>) -> Result<Option<Box<dyn ArrayReader>>> {
        let field = args.field;
        let (col_idx, primitive_type, vector_length) = match &field.field_type {
            ParquetFieldType::Primitive {
                col_idx,
                primitive_type,
                vector_length,
            } => match primitive_type.as_ref() {
                Type::PrimitiveType { .. } => (*col_idx, primitive_type.clone(), *vector_length),
                Type::GroupType { .. } => unreachable!(),
            },
            _ => unreachable!(),
        };

        if !args.mask.leaf_included(col_idx) {
            return Ok(None);
        }

        let physical_type = primitive_type.get_physical_type();

        // We don't track the column path in ParquetField as it adds a potential source
        // of bugs when the arrow mapping converts more than one level in the parquet
        // schema into a single arrow field.
        //
        // None of the readers actually use this field, but it is required for this type,
        // so just stick a placeholder in
        let column_desc = Arc::new(ColumnDescriptor::new_with_repeated_ancestor_and_vector(
            primitive_type,
            field.def_level,
            field.rep_level,
            ColumnPath::new(vec![]),
            0,
            vector_length,
        ));

        let mut page_iterator = self.row_groups.column_chunks(col_idx)?;
        if let Some(vector_length) = vector_length {
            validate_vector_column_chunks(self.row_groups.row_groups(), col_idx, vector_length)?;
            let vector_length = usize::try_from(vector_length).map_err(|_| {
                general_err!(
                    "VECTOR column {} has invalid vector_length {}",
                    col_idx,
                    vector_length
                )
            })?;
            page_iterator = Box::new(VectorPageIterator::new(
                page_iterator,
                col_idx,
                vector_length,
            ));
        }

        let arrow_type = Some(field.arrow_type.clone());

        // LogicalType::Unknown maps to DataType::Null. In the past it has been assumed
        // that only INT32 can have this annotation, but this is not required by the Parquet
        // specification. Since this can only annotate an entirely null column, the data type
        // used for the NullArrayReader should be irrelevant. It's just needed to read the
        // repetition and definition level data.
        if matches!(arrow_type, Some(DataType::Null)) {
            // A VECTOR element must never resolve to Null (a dense vector of nulls
            // is contradictory). Schema conversion already rejects this; guard
            // here too so the NullArrayReader short-circuit can't bypass the VECTOR
            // reshape wrapper in the parent FixedSizeList reader.
            if vector_length.is_some() {
                return Err(general_err!(
                    "VECTOR column with a null/unknown element type is not supported"
                ));
            }
            let reader = Box::new(NullArrayReader::<Int32Type>::new(
                page_iterator,
                column_desc,
                self.batch_size,
            )?) as _;
            return Ok(Some(reader));
        }

        let reader = match physical_type {
            PhysicalType::BOOLEAN => Box::new(PrimitiveArrayReader::<BoolType>::new(
                page_iterator,
                column_desc,
                arrow_type,
                self.batch_size,
            )?) as _,
            PhysicalType::INT32 => Box::new(PrimitiveArrayReader::<Int32Type>::new(
                page_iterator,
                column_desc,
                arrow_type,
                self.batch_size,
            )?) as _,
            PhysicalType::INT64 => Box::new(PrimitiveArrayReader::<Int64Type>::new(
                page_iterator,
                column_desc,
                arrow_type,
                self.batch_size,
            )?) as _,
            PhysicalType::INT96 => Box::new(PrimitiveArrayReader::<Int96Type>::new(
                page_iterator,
                column_desc,
                arrow_type,
                self.batch_size,
            )?) as _,
            PhysicalType::FLOAT => Box::new(PrimitiveArrayReader::<FloatType>::new(
                page_iterator,
                column_desc,
                arrow_type,
                self.batch_size,
            )?) as _,
            PhysicalType::DOUBLE => Box::new(PrimitiveArrayReader::<DoubleType>::new(
                page_iterator,
                column_desc,
                arrow_type,
                self.batch_size,
            )?) as _,
            PhysicalType::BYTE_ARRAY => match arrow_type {
                Some(DataType::Dictionary(_, _)) => make_byte_array_dictionary_reader(
                    page_iterator,
                    column_desc,
                    arrow_type,
                    self.batch_size,
                )?,
                Some(DataType::Utf8View | DataType::BinaryView) => make_byte_view_array_reader(
                    page_iterator,
                    column_desc,
                    arrow_type,
                    self.batch_size,
                )?,
                _ => {
                    make_byte_array_reader(page_iterator, column_desc, arrow_type, self.batch_size)?
                }
            },
            PhysicalType::FIXED_LEN_BYTE_ARRAY => match arrow_type {
                Some(DataType::Dictionary(_, _)) => make_byte_array_dictionary_reader(
                    page_iterator,
                    column_desc,
                    arrow_type,
                    self.batch_size,
                )?,
                _ => make_fixed_len_byte_array_reader(
                    page_iterator,
                    column_desc,
                    arrow_type,
                    self.batch_size,
                )?,
            },
        };

        Ok(Some(reader))
    }

    fn build_struct_reader(&self, args: ReaderArgs<'_>) -> Result<Option<Box<dyn ArrayReader>>> {
        let field = args.field;
        let arrow_fields = match &field.arrow_type {
            DataType::Struct(children) => children,
            _ => unreachable!(),
        };
        let children = field.children().unwrap();
        assert_eq!(arrow_fields.len(), children.len());

        let mut readers = Vec::with_capacity(children.len());
        let mut builder = SchemaBuilder::with_capacity(children.len());

        for (arrow, parquet) in arrow_fields.iter().zip(children) {
            if let Some(reader) = self.build_reader(args.with_field(parquet))? {
                // Need to retrieve underlying data type to handle projection
                let child_type = reader.get_data_type().clone();
                builder.push(arrow.as_ref().clone().with_data_type(child_type));
                readers.push(reader);
            }
        }

        if readers.is_empty() {
            return Ok(None);
        }

        Ok(Some(Box::new(StructArrayReader::new(
            DataType::Struct(builder.finish().fields),
            readers,
            field.def_level,
            field.rep_level,
            field.nullable,
        ))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::schema::ArrowSchemaConverter;
    use crate::arrow::schema::parquet_to_arrow_schema_and_fields;
    use crate::arrow::schema::virtual_type::RowNumber;
    use crate::basic::Encoding;
    use crate::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};
    use crate::file::reader::{FileReader, SerializedFileReader};
    use crate::util::test_common::file_util::get_test_file;
    use arrow::datatypes::{Field, Schema};
    use bytes::Bytes;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn test_create_array_reader() {
        let file = get_test_file("nulls.snappy.parquet");
        let file_reader: Arc<dyn FileReader> = Arc::new(SerializedFileReader::new(file).unwrap());

        let file_metadata = file_reader.metadata().file_metadata();
        let mask = ProjectionMask::leaves(file_metadata.schema_descr(), [0]);
        let (_, fields) = parquet_to_arrow_schema_and_fields(
            file_metadata.schema_descr(),
            ProjectionMask::all(),
            file_metadata.key_value_metadata(),
            &[],
        )
        .unwrap();

        let metrics = ArrowReaderMetrics::disabled();
        let array_reader = ArrayReaderBuilder::new(&file_reader, &metrics)
            .with_batch_size(DEFAULT_BATCH_SIZE)
            .build_array_reader(fields.as_ref(), &mask)
            .unwrap();

        // Create arrow types
        let arrow_type = DataType::Struct(Fields::from(vec![Field::new(
            "b_struct",
            DataType::Struct(vec![Field::new("b_c_int", DataType::Int32, true)].into()),
            true,
        )]));

        assert_eq!(array_reader.get_data_type(), &arrow_type);
    }

    fn vector_row_group(num_rows: i64, num_values: i64) -> RowGroupMetaData {
        let arrow_schema = Schema::new(vec![Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("element", DataType::Float32, false)), 3),
            false,
        )]);
        let schema_descr = Arc::new(
            ArrowSchemaConverter::new()
                .with_vector_encoding(true)
                .convert(&arrow_schema)
                .unwrap(),
        );
        let column = ColumnChunkMetaData::builder(schema_descr.column(0))
            .set_num_values(num_values)
            .build()
            .unwrap();
        RowGroupMetaData::builder(schema_descr)
            .set_num_rows(num_rows)
            .set_column_metadata(vec![column])
            .build()
            .unwrap()
    }

    #[test]
    fn vector_column_chunk_metadata_must_match_row_count() {
        let valid = vector_row_group(2, 6);
        validate_vector_column_chunks([&valid], 0, 3).unwrap();

        let invalid = vector_row_group(2, 5);
        let err = validate_vector_column_chunks([&invalid], 0, 3)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("num_values 5") && err.contains("expected 6"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn vector_page_metadata_must_match_row_count() {
        let valid = PageMetadata {
            num_rows: Some(2),
            num_levels: Some(6),
            num_nulls: Some(0),
            is_dict: false,
        };
        validate_vector_page_metadata(0, 0, &valid, 3).unwrap();

        let invalid = PageMetadata {
            num_rows: Some(2),
            num_levels: Some(5),
            num_nulls: Some(0),
            is_dict: false,
        };
        let err = validate_vector_page_metadata(0, 0, &invalid, 3)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("num_values 5") && err.contains("vector_length 3"),
            "unexpected error: {err}"
        );

        let sparse = PageMetadata {
            num_rows: Some(2),
            num_levels: Some(6),
            num_nulls: Some(1),
            is_dict: false,
        };
        let err = validate_vector_page_metadata(0, 0, &sparse, 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("null values"), "unexpected error: {err}");
    }

    fn vector_data_page_v2(num_nulls: u32) -> Page {
        Page::DataPageV2 {
            buf: Bytes::new(),
            num_values: 6,
            encoding: Encoding::PLAIN,
            num_nulls,
            num_rows: 2,
            def_levels_byte_len: 0,
            rep_levels_byte_len: 0,
            is_compressed: false,
            statistics: None,
        }
    }

    #[test]
    fn vector_data_page_v2_must_be_dense() {
        let err = validate_vector_page(0, 0, &vector_data_page_v2(1), 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("null values"), "unexpected error: {err}");
    }

    struct MockPageReader {
        metadata: Option<PageMetadata>,
        page: Option<Page>,
        get_called: Arc<AtomicBool>,
        skip_called: Arc<AtomicBool>,
    }

    impl Iterator for MockPageReader {
        type Item = Result<Page>;

        fn next(&mut self) -> Option<Self::Item> {
            self.get_next_page().transpose()
        }
    }

    impl PageReader for MockPageReader {
        fn get_next_page(&mut self) -> Result<Option<Page>> {
            self.get_called.store(true, Ordering::SeqCst);
            Ok(self.page.take())
        }

        fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
            Ok(self.metadata.clone())
        }

        fn skip_next_page(&mut self) -> Result<()> {
            self.skip_called.store(true, Ordering::SeqCst);
            self.metadata = None;
            self.page = None;
            Ok(())
        }
    }

    fn skip_vector_page(metadata: PageMetadata, page: Option<Page>) -> (String, bool, bool) {
        let get_called = Arc::new(AtomicBool::new(false));
        let skip_called = Arc::new(AtomicBool::new(false));
        let inner = MockPageReader {
            metadata: Some(metadata),
            page,
            get_called: Arc::clone(&get_called),
            skip_called: Arc::clone(&skip_called),
        };
        let mut reader = VectorPageReader {
            inner: Box::new(inner),
            col_idx: 0,
            vector_length: 3,
            row_group_idx: 0,
        };

        let err = reader.skip_next_page().unwrap_err().to_string();
        (
            err,
            get_called.load(Ordering::SeqCst),
            skip_called.load(Ordering::SeqCst),
        )
    }

    #[test]
    fn vector_skip_next_page_validates_dense_pages() {
        let (err, get_called, skip_called) = skip_vector_page(
            PageMetadata {
                num_rows: Some(2),
                num_levels: Some(6),
                num_nulls: Some(1),
                is_dict: false,
            },
            None,
        );
        assert!(err.contains("null values"), "unexpected error: {err}");
        assert!(!get_called);
        assert!(!skip_called);

        let (err, get_called, skip_called) = skip_vector_page(
            PageMetadata {
                num_rows: Some(2),
                num_levels: None,
                num_nulls: None,
                is_dict: false,
            },
            Some(vector_data_page_v2(1)),
        );
        assert!(err.contains("null values"), "unexpected error: {err}");
        assert!(get_called);
        assert!(!skip_called);
    }

    #[test]
    fn test_create_array_reader_with_row_numbers() {
        let file = get_test_file("nulls.snappy.parquet");
        let file_reader: Arc<dyn FileReader> = Arc::new(SerializedFileReader::new(file).unwrap());

        let file_metadata = file_reader.metadata().file_metadata();
        let mask = ProjectionMask::leaves(file_metadata.schema_descr(), [0]);
        let row_number_field = Arc::new(
            Field::new("row_number", DataType::Int64, false).with_extension_type(RowNumber),
        );
        let (_, fields) = parquet_to_arrow_schema_and_fields(
            file_metadata.schema_descr(),
            ProjectionMask::all(),
            file_metadata.key_value_metadata(),
            std::slice::from_ref(&row_number_field),
        )
        .unwrap();

        let metrics = ArrowReaderMetrics::disabled();
        let array_reader = ArrayReaderBuilder::new(&file_reader, &metrics)
            .with_batch_size(DEFAULT_BATCH_SIZE)
            .with_parquet_metadata(file_reader.metadata())
            .build_array_reader(fields.as_ref(), &mask)
            .unwrap();

        // Create arrow types
        let arrow_type = DataType::Struct(Fields::from(vec![
            Field::new(
                "b_struct",
                DataType::Struct(vec![Field::new("b_c_int", DataType::Int32, true)].into()),
                true,
            ),
            (*row_number_field).clone(),
        ]));

        assert_eq!(array_reader.get_data_type(), &arrow_type);
    }
}
