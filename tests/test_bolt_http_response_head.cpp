#include <gtest/gtest.h>
#include <string>
#include "bolt/net/bolt_http_response_head.h"
namespace bn=bolt::net;
static bn::HttpHeadParse parse(const std::string& input,bn::HttpResponseHead& out) {
    return bn::http_response_head_parse(input.data(),static_cast<std::uint32_t>(input.size()),&out);
}
TEST(BoltHttpResponseHead, EveryTruncationAndBorrowedFields) {
    const std::string valid="HTTP/1.1 200 OK\r\nContent-Length: 12\r\nX-Test: value\r\n\r\n";
    bn::HttpResponseHead out{};
    for (std::size_t n=0;n<valid.size();++n) EXPECT_EQ(bn::http_response_head_parse(valid.data(),static_cast<std::uint32_t>(n),&out),bn::HttpHeadParse::kIncomplete)<<n;
    const auto with_body=valid+"body";
    ASSERT_EQ(parse(with_body,out),bn::HttpHeadParse::kComplete);
    EXPECT_EQ(out.status,200u); EXPECT_EQ(out.bytes,valid.size()); EXPECT_EQ(out.content_length,12u);
    EXPECT_TRUE(out.has_content_length); EXPECT_FALSE(out.has_transfer_encoding);
    EXPECT_EQ(std::string(out.headers[1].value.bytes,out.headers[1].value.size),"value");
}
TEST(BoltHttpResponseHead, MalformedAndAmbiguousRefuse) {
    const char* bad[]={
        "HTTP/2.0 200 OK\r\n\r\n","HTTP/1.1 20 OK\r\n\r\n","HTTP/1.1 600 OK\r\n\r\n",
        "HTTP/1.1 200 OK\n\n","HTTP/1.1 200 OK\rX\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Length: 1\r\ncontent-length: 1\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n",
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 1\r\n\r\n",
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: gzip\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Length: 18446744073709551616\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\n","HTTP/1.1 200 OK\r\nBad Name: 1\r\n\r\n",
        "HTTP/1.1 200 OK\r\n folded\r\n\r\n"};
    bn::HttpResponseHead out{};
    for (const auto* value:bad) EXPECT_EQ(parse(value,out),bn::HttpHeadParse::kInvalid)<<value;
    EXPECT_EQ(parse(std::string(bn::kHttpResponseHeadCap,'x'),out),bn::HttpHeadParse::kInvalid);
    std::string many="HTTP/1.0 200 OK\r\n";
    for (auto i=0u;i<=bn::kHttpResponseHeaderCap;++i) many+="X: y\r\n";
    EXPECT_EQ(parse(many+"\r\n",out),bn::HttpHeadParse::kInvalid);
    EXPECT_EQ(parse("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",out),bn::HttpHeadParse::kComplete);
    EXPECT_TRUE(out.has_transfer_encoding); // caller decides supported framing
}
