// bolt/ingest/bolt_parquet_nested.h — nested STRUCT / LIST / MAP read path.
//
// PqMeta keeps only LEAF columns. This header adds the GROUP tree the leaves
// hang off, in two views, and a reader that assembles one top-level field of
// a row group into a ColumnFormat::Nested BoltColumn tree.
//
//   PqSchemaNode  one raw parquet SchemaElement, DFS order, [0] = root.
//                 Names, repetition, ConvertedType/LogicalType, field id
//                 (SchemaElement field 9) on EVERY node, groups included.
//   PqField       the Bolt-shaped interpretation of that tree: Struct, List,
//                 Map or Leaf, one PqField per BoltColumn the reader emits,
//                 in the same parent/child order. Field ids live here too,
//                 taken from the parquet node that carries the logical field:
//                   struct field / leaf      its own node
//                   list (LIST group)        the LIST group
//                   list element             3-level: the element node;
//                                            2-level: the repeated node
//                   map (MAP group)          the MAP group
//                   map entry struct         the repeated key_value node
//                   map key / value          the key / value nodes
//                 So an Iceberg reader resolves nested columns by id with
//                 pq_field_find_id, never by name.
//
// LIST/MAP shapes follow LogicalTypes.md including backward compatibility:
//   - group (LIST) { repeated group <any> { <elem> } }          3-level
//     (pyarrow list.element / list.item, Spark list.element, bag.array)
//   - group (LIST) { repeated <leaf> }                         2-level
//   - group (LIST) { repeated group with >1 field }            element = it
//   - group (LIST) { repeated group array|<name>_tuple {one} } element = it
//   - repeated field outside any LIST group                    required list
//                                                              of required
//   - group (MAP | MAP_KEY_VALUE) { repeated group { key; value } }
// A MAP whose repeated group lacks a value field is refused (Bolt Map is a
// list of 2-field structs).
//
// Assembly semantics (all distinguished): null struct vs struct whose fields
// are null; null list vs empty list vs list with null elements. A field
// under a null struct is itself null in its child column (rows stay aligned).

#pragma once

#include <cstdint>

#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_nested_limits.h"

namespace bolt {

class Arena;
struct BoltColumn;

namespace ingest {
namespace parquet {

enum class PqFieldKind : uint8_t { Leaf = 0, Struct = 1, List = 2, Map = 3 };

struct PqSchemaNode {
    char     name[kPqMaxNameBytes];   // this element's own name, not dotted
    int32_t  field_id;                // valid iff has_field_id
    int32_t  converted;               // ConvertedType, -1 = none
    int32_t  logical;                 // PqLogical, -1 = none (List / Map too)
    int32_t  parent;                  // -1 for the root
    int32_t  first_child;             // -1 for a leaf
    int32_t  next_sibling;            // -1 for the last child
    int32_t  column;                  // leaf: index into PqMeta::columns; -1
    uint32_t n_children;
    uint32_t col_begin;               // leaf columns of the subtree:
    uint32_t col_end;                 //   [col_begin, col_end)
    uint8_t  has_field_id;
    uint8_t  repetition;              // 0 REQUIRED / 1 OPTIONAL / 2 REPEATED
    uint8_t  def;                     // definition level reached AT this node
    uint8_t  rep;                     // repetition level reached AT this node
};

struct PqField {
    char     name[kPqMaxNameBytes];
    int32_t  field_id;                // valid iff has_field_id
    int32_t  node;                    // PqSchemaNode the field is named after
    int32_t  column;                  // Leaf: PqMeta::columns index; else -1
    int32_t  first_child;             // Struct: fields; List: element;
    int32_t  next_sibling;            // Map: the entry Struct {key, value}
    uint32_t n_children;
    uint32_t col_begin;               // leaf columns under this field
    uint32_t col_end;
    PqFieldKind kind;
    uint8_t  has_field_id;
    uint8_t  nullable;                // the value itself may be null
    // Dremel thresholds the assembler applies (levels of the leaf streams):
    //   a slot is an instance of this field iff def >= inst_def and
    //   rep <= inst_rep; the instance is non-null iff def >= valid_def;
    //   List/Map: a slot carries an element iff def >= elem_def and
    //   rep <= elem_rep.
    uint8_t  inst_def;
    uint8_t  inst_rep;
    uint8_t  valid_def;
    uint8_t  elem_def;
    uint8_t  elem_rep;
};

struct PqSchemaTree {
    PqSchemaNode* nodes;              // arena-owned, n_nodes entries
    PqField*      fields;             // arena-owned, n_fields entries
    int32_t*      top;                // top-level field indices, schema order
    uint32_t      n_nodes;
    uint32_t      n_fields;
    uint32_t      n_top;
    uint32_t      n_columns;          // leaves; equals PqMeta::n_columns
};

// Parse the schema tree from a whole file (footer located like
// parquet_read_meta) or from a FileMetaData slice. All storage from `arena`.
// False on corrupt input, a shape outside the rules above, depth past
// kPqMaxNestDepth or more than kPqMaxSchemaElems elements.
bool parquet_read_schema_tree(const uint8_t* buf, uint64_t len, Arena* arena,
                              PqSchemaTree* out) noexcept;
bool parquet_schema_tree_slice(const uint8_t* metadata, uint32_t length,
                               Arena* arena, PqSchemaTree* out) noexcept;

// Child of `parent` (a field index, or -1 for the top level) carrying
// `field_id`; -1 when absent. For a List the only child is the element; for
// a Map the entry Struct, whose children are key and value.
int32_t pq_field_find_id(const PqSchemaTree* tree, int32_t parent,
                         int32_t field_id) noexcept;

// Decode every leaf of top-level field `top_field` (index into tree->top) in
// `row_group` and assemble it. A Leaf top-level field returns the ordinary
// flat column, so callers can use one entry point for every column. The
// emitted tree mirrors the PqField tree: child i of a Struct is the i-th
// PqField child; a List's child 0 is its element; a Map's child 0 is the
// entry Struct {key, value} (BoltType::Map, list offsets as for List).
// `meta` must be the same file's footer; the tree is checked against it.
bool parquet_read_nested_column_tree(const uint8_t* buf, uint64_t len,
                                     const PqMeta* meta,
                                     const PqSchemaTree* tree,
                                     uint32_t row_group, uint32_t top_field,
                                     Arena* arena, BoltColumn* out,
                                     int64_t* out_rows) noexcept;

// Same, parsing the schema tree from `buf`'s footer on each call.
bool parquet_read_nested_column(const uint8_t* buf, uint64_t len,
                                const PqMeta* meta, uint32_t row_group,
                                uint32_t top_field, Arena* arena,
                                BoltColumn* out, int64_t* out_rows) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
