#include "bolt/lakehouse/iceberg/default_value.h"
#include <gtest/gtest.h>
#include <cstring>
#include <string>
namespace {
using namespace bolt;using namespace bolt::lakehouse::iceberg;using S=JsonValueStatus;using M=DefaultValueMode;
SchemaField field(int32_t id,const char* name,const char* type,bool required=false) {
    SchemaField out{};out.id=id;out.required=required;std::strcpy(out.name,name);std::strcpy(out.type,type);return out;
}
void defaults(Arena& a,SchemaField& f,const char* initial,const char* write) {
    ASSERT_TRUE(schema_default_assign(&f.initial_default_value,reinterpret_cast<const uint8_t*>(initial),static_cast<uint32_t>(std::strlen(initial)),&a));
    ASSERT_TRUE(schema_default_assign(&f.write_default_value,reinterpret_cast<const uint8_t*>(write),static_cast<uint32_t>(std::strlen(write)),&a));
}
std::string emit(Arena& a,const SchemaField& f,const BoltColumn& c) {
    const uint8_t* json=nullptr;uint32_t size=0;EXPECT_EQ(json_value_encode(f,c,0,&a,&json,&size),S::Ok);
    return json?std::string(reinterpret_cast<const char*>(json),size):"";
}
TEST(IcebergDefaults, EmptyDeclarationUsesChildMetadataInTheChosenMode) {
    Arena a;auto root=field(1,"point","struct");SchemaField children[]{field(2,"x","long",true),field(3,"inner","struct")};
    auto grandchild=field(4,"n","long",true);children[1].children=&grandchild;children[1].n_children=1;root.children=children;root.n_children=2;
    defaults(a,root,"{}","{}");defaults(a,children[0],"7","9");defaults(a,children[1],"null","{}");defaults(a,grandchild,"11","13");
    Schema schema{};schema.fields=&root;schema.n_fields=1;EXPECT_EQ(schema_defaults_validate(schema,&a),S::Ok);
    BoltColumn value{};ASSERT_EQ(default_value_decode(root,M::Initial,&a,&value),S::Ok);
    EXPECT_EQ(*static_cast<const int64_t*>(value.child_at(0)->data),7);EXPECT_TRUE(value.child_at(1)->is_null(0));
    EXPECT_EQ(emit(a,root,value),R"({"2":7})");
    ASSERT_EQ(default_value_decode(root,M::Write,&a,&value),S::Ok);
    EXPECT_EQ(*static_cast<const int64_t*>(value.child_at(0)->data),9);EXPECT_EQ(*static_cast<const int64_t*>(value.child_at(1)->child_at(0)->data),13);
    defaults(a,root,"null","null");ASSERT_EQ(default_value_decode(root,M::Write,&a,&value),S::Ok);
    EXPECT_TRUE(value.is_null(0));EXPECT_TRUE(value.child_at(0)->is_null(0));EXPECT_TRUE(value.child_at(1)->child_at(0)->is_null(0));
}
TEST(IcebergDefaults, NonemptyStoredStructAndProhibitedTypesRefuseTransactionally) {
    Arena a;auto root=field(1,"point","struct");auto child=field(2,"n","long",true);root.children=&child;root.n_children=1;
    defaults(a,child,"7","9");defaults(a,root,R"({"2":7})","{}");BoltColumn value{};value.length=919;const auto before=value;
    EXPECT_EQ(default_value_decode(root,M::Initial,&a,&value),S::Invalid);EXPECT_EQ(std::memcmp(&value,&before,sizeof(value)),0);
    Schema schema{};schema.fields=&root;schema.n_fields=1;EXPECT_EQ(schema_defaults_validate(schema,&a),S::Invalid);
    for(const char* type:{"unknown","variant","geometry","geography"}) {
        root=field(1,"v",type);defaults(a,root,"null","\"POINT (1 2)\"");EXPECT_EQ(default_value_decode(root,M::Write,&a,&value),S::Invalid);
    }
    root=field(1,"required","long",true);defaults(a,root,"null","null");EXPECT_EQ(schema_defaults_validate(schema,&a),S::Ok);
    EXPECT_EQ(default_value_decode(root,M::Write,&a,&value),S::Invalid);
}
TEST(IcebergDefaults, RebindPreservesMissingCollectionFieldsAndLateDefaultChanges) {
    Arena a;auto root=field(1,"list","list"),element=field(2,"element","struct");SchemaField children[]{field(3,"n","long",true),field(4,"optional","long")};
    element.children=children;element.n_children=2;root.children=&element;root.n_children=1;defaults(a,children[0],"3","5");
    defaults(a,root,"[]",R"([{}, {"3":7}, null])");Schema source{};source.fields=&root;source.n_fields=1;Schema target{};
    ASSERT_TRUE(schema_copy(&target,&source,&a,nullptr));auto& mapped=target.fields[0];mapped.id=10;mapped.children[0].id=20;
    mapped.children[0].children[0].id=30;mapped.children[0].children[1].id=40;SchemaDefault rebound{};
    ASSERT_EQ(default_value_rebind(root,mapped,M::Write,&a,&rebound),S::Ok);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(rebound.json),rebound.size),R"([{},{"30":7},null])");
    mapped.write_default_value=rebound;defaults(a,mapped.children[0].children[0],"3","9");BoltColumn value{};
    ASSERT_EQ(default_value_decode(mapped,M::Write,&a,&value),S::Ok);const auto* elements=value.list_element();
    EXPECT_EQ(static_cast<const int64_t*>(elements->child_at(0)->data)[0],9);
    EXPECT_EQ(static_cast<const int64_t*>(elements->child_at(0)->data)[1],7);EXPECT_TRUE(elements->is_null(2));
}
TEST(IcebergDefaults, StandaloneSchemaOwnsDefaultsAndRejectsDuplicateAliases) {
    Arena a;std::string json=R"({"type":"struct","schema-id":7,"fields":[{"id":1,"name":"s","type":"struct","write-default":{},"type":{"type":"struct","fields":[]}}]})";
    Schema schema{};schema.schema_id=919;EXPECT_FALSE(schema_json_parse(reinterpret_cast<const uint8_t*>(json.data()),static_cast<uint32_t>(json.size()),&a,&schema));EXPECT_EQ(schema.schema_id,919);
    json=R"({"type":"struct","schema-id":7,"fields":[{"id":1,"name":"s","type":{"type":"struct","fields":[]},"write-default":{}}]})";
    ASSERT_TRUE(schema_json_parse(reinterpret_cast<const uint8_t*>(json.data()),static_cast<uint32_t>(json.size()),&a,&schema));json.assign(json.size(),'x');
    EXPECT_EQ(schema.schema_id,7);EXPECT_STREQ(schema.fields[0].name,"s");EXPECT_EQ(schema.fields[0].write_default_value.size,2u);
    for(const char* bad:{R"({"fields":[],"\u0066ields":[]})",R"({"type":"list","fields":[]})",R"({"schema-id":1})",R"({"fields":[]} trailing)"})
        EXPECT_FALSE(schema_json_parse(reinterpret_cast<const uint8_t*>(bad),static_cast<uint32_t>(std::strlen(bad)),&a,&schema));
}
TEST(IcebergDefaults, CapacityLimitsRefuseWithoutPublishing) {
    Arena a;auto root=field(1,"s","struct");auto child=field(2,"n","long");root.children=&child;root.n_children=1;
    defaults(a,root,"{}","{}");defaults(a,child,"7","9");BoltColumn value{};value.length=919;
    JsonValueLimits limits{};limits.values=1;EXPECT_EQ(default_value_decode(root,M::Write,&a,&value,limits),S::Capacity);EXPECT_EQ(value.length,919);
    limits={};limits.json_bytes=1;EXPECT_EQ(default_value_decode(root,M::Write,&a,&value,limits),S::Capacity);EXPECT_EQ(value.length,919);
}
TEST(IcebergDefaults, InitialEqualityUsesTypedPromotionsAndDecodedText) {
    Arena a;auto before=field(1,"old","int"),after=field(1,"renamed","long");
    defaults(a,before,"7","8");defaults(a,after,"7","99");bool same=false;
    ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_TRUE(same);
    defaults(a,after,"8","99");ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_FALSE(same);
    before=field(1,"f","float");after=field(1,"f","double");defaults(a,before,"0.1","0.1");
    defaults(a,after,"0.10000000149011612","0.2");ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_TRUE(same);
    defaults(a,after,"0.1","0.2");ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_FALSE(same);
    before=field(1,"d","decimal(9,2)");after=field(1,"d","decimal(30,2)");
    defaults(a,before,"\"1.20\"","null");defaults(a,after,"\"01.20\"","null");
    ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_TRUE(same);
    before=field(1,"s","string");after=before;defaults(a,before,"\"a\\u0062\"","null");defaults(a,after,"\"ab\"","null");
    ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_TRUE(same);
}
TEST(IcebergDefaults, InitialEqualityProjectsStableChildrenAcrossReorderAddAndDrop) {
    Arena a;auto before=field(1,"items","list"),after=before;
    auto old_element=field(2,"element","struct"),new_element=old_element;
    SchemaField old_children[]{field(3,"x","int"),field(4,"s","string")};
    SchemaField new_children[]{field(5,"added","long"),field(4,"renamed","string"),field(3,"x","long")};
    defaults(a,old_children[0],"7","8");defaults(a,new_children[2],"7","9");defaults(a,new_children[0],"11","13");
    old_element.children=old_children;old_element.n_children=2;new_element.children=new_children;new_element.n_children=3;
    before.children=&old_element;before.n_children=1;after.children=&new_element;after.n_children=1;
    defaults(a,before,R"([{"4":"ab"},null])","[]");defaults(a,after,R"([{"3":7,"4":"a\u0062","5":11},null])","[]");
    bool same=false;ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_TRUE(same);
    Schema source{},target{};source.fields=&before;source.n_fields=1;target.fields=&after;target.n_fields=1;
    EXPECT_EQ(schema_initial_defaults_validate(source,target,&a),S::Ok);
    defaults(a,new_children[2],"8","9");EXPECT_EQ(schema_initial_defaults_validate(source,target,&a),S::Invalid);
    defaults(a,new_children[2],"7","9");new_element.children=new_children+1;new_element.n_children=1;
    defaults(a,after,R"([{"4":"ab"},null])","[]");EXPECT_EQ(schema_initial_defaults_validate(source,target,&a),S::Ok);
}
TEST(IcebergDefaults, InitialMapEqualityIgnoresEntryOrderButProtectsKeysAndValues) {
    Arena a;auto before=field(1,"m","map"),after=before;
    SchemaField children[]{field(2,"key","string",true),field(3,"value","long")};
    before.children=children;before.n_children=2;after=before;
    defaults(a,before,R"({"keys":["a","b"],"values":[7,null]})","null");
    defaults(a,after,R"({"values":[null,7],"keys":["b","\u0061"]})","null");
    bool same=false;ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_TRUE(same);
    defaults(a,after,R"({"values":[7,null],"keys":["b","a"]})","null");
    ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_FALSE(same);
}
TEST(IcebergDefaults, InitialNullMaskDoesNotHideChangedChildDeclaration) {
    Arena a;auto before=field(1,"point","struct"),after=before;
    auto old_child=field(2,"x","long",true),new_child=old_child;
    defaults(a,old_child,"7","8");defaults(a,new_child,"9","10");before.children=&old_child;before.n_children=1;
    after.children=&new_child;after.n_children=1;defaults(a,before,"null","{}");defaults(a,after,"null","{}");
    bool same=false;ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_TRUE(same);
    Schema source{},target{};source.fields=&before;source.n_fields=1;target.fields=&after;target.n_fields=1;
    EXPECT_EQ(schema_initial_defaults_validate(source,target,&a),S::Invalid);
    defaults(a,new_child,"7","10");EXPECT_EQ(schema_initial_defaults_validate(source,target,&a),S::Ok);
    before=field(1,"required","long",true);after=before;defaults(a,after,"null","null");
    ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_TRUE(same);
    JsonValueLimits limits{};limits.values=1;before=field(1,"point","struct");before.children=&old_child;before.n_children=1;
    defaults(a,before,"{}","{}");same=true;
    EXPECT_EQ(initial_default_equal(before,before,&a,&same,limits),S::Capacity);EXPECT_TRUE(same);
}
TEST(IcebergDefaults, InitialComplexMapKeysAreCanonicalAndShapeChangesRefuse) {
    Arena a;auto before=field(1,"m","map"),after=before;
    SchemaField old_parts[]{field(2,"key","struct",true),field(3,"value","long")};
    SchemaField new_parts[]{old_parts[0],old_parts[1]};
    SchemaField old_key[]{field(4,"n","int"),field(5,"s","string")};
    SchemaField new_key[]{field(5,"renamed","string"),field(4,"n","long")};
    old_parts[0].children=old_key;old_parts[0].n_children=2;new_parts[0].children=new_key;new_parts[0].n_children=2;
    before.children=old_parts;before.n_children=2;after.children=new_parts;after.n_children=2;
    defaults(a,before,R"({"keys":[{"4":1,"5":"a"},{"4":2}],"values":[7,8]})","null");
    defaults(a,after,R"({"keys":[{"4":2},{"5":"\u0061","4":1}],"values":[8,7]})","null");
    bool same=false;ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_TRUE(same);
    new_parts[0].n_children=1;defaults(a,after,R"({"keys":[{"5":"a"},{"5":"b"}],"values":[7,8]})","null");
    EXPECT_EQ(initial_default_equal(before,after,&a,&same),S::Invalid);EXPECT_TRUE(same);
    before=field(1,"f","double");after=before;defaults(a,before,"-0.0","null");defaults(a,after,"0.0","null");
    ASSERT_EQ(initial_default_equal(before,after,&a,&same),S::Ok);EXPECT_FALSE(same);
}
} // namespace
