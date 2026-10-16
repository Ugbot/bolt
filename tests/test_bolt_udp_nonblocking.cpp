#include <gtest/gtest.h>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstring>
#include <thread>
#include <type_traits>
#include "bolt/net/bolt_udp.h"
namespace bn=bolt::net;
TEST(BoltUdpNonblocking, ClosedEndpointRefusesModeChange) {
    bn::UdpEndpoint endpoint{};endpoint.sock=bn::k_udp_invalid_sock;
    EXPECT_FALSE(bn::udp_endpoint_set_nonblocking(&endpoint,true));
    EXPECT_FALSE(bn::udp_endpoint_set_nonblocking(&endpoint,false));
}
TEST(BoltUdpNonblocking, EmptyReceiveReturnsAndDatagramsStillArrive) {
    bn::UdpEndpoint receiver{},sender{};
    ASSERT_TRUE(bn::udp_endpoint_open(&receiver,0u,500u));
    ASSERT_TRUE(bn::udp_endpoint_open(&sender,0u,500u));
    ASSERT_TRUE(bn::udp_endpoint_set_nonblocking(&receiver,true));
    const auto began=std::chrono::steady_clock::now();
    std::uint32_t value=0u;bn::UdpAddr from{};
    EXPECT_EQ(bn::udp_recv_from(&receiver,&value,sizeof(value),&from),-1);
    EXPECT_LT(std::chrono::steady_clock::now()-began,std::chrono::milliseconds(100));
    const std::uint32_t sent=123u;
    ASSERT_EQ(bn::udp_send_to(&sender,{bn::k_udp_loopback_ipv4,receiver.local_port,0u},&sent,sizeof(sent)),sizeof(sent));
    std::int64_t length=-1;
    for (unsigned i=0;i<100u && length<0;++i) {
        length=bn::udp_recv_from(&receiver,&value,sizeof(value),&from);
        if (length<0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(length,sizeof(value));EXPECT_EQ(value,sent);EXPECT_EQ(from.port,sender.local_port);
    ASSERT_TRUE(bn::udp_endpoint_set_nonblocking(&receiver,false));
    std::thread delayed([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_EQ(bn::udp_send_to(&sender,{bn::k_udp_loopback_ipv4,receiver.local_port,0u},&sent,sizeof(sent)),sizeof(sent));
    });
    const auto blocked=bn::udp_recv_from(&receiver,&value,sizeof(value),&from);delayed.join();
    EXPECT_EQ(blocked,sizeof(value));
    bn::udp_endpoint_close(&receiver);bn::udp_endpoint_close(&sender);
}

static_assert(std::is_standard_layout_v<bn::UdpReceiveResult>);
static_assert(std::is_trivially_copyable_v<bn::UdpReceiveResult>);

namespace {
constexpr unsigned kReceiveAttempts = 200u;
constexpr bn::UdpAddr kPeerSentinel{0x01020304u, 0x5678u, 0x9ABCu};

class BoltUdpChecked : public ::testing::Test {
protected:
    bn::UdpEndpoint receiver{bn::k_udp_invalid_sock, 0u, {}};
    bn::UdpEndpoint sender{bn::k_udp_invalid_sock, 0u, {}};

    void SetUp() override {
        ASSERT_TRUE(bn::udp_endpoint_open(&receiver, 0u, 500u));
        ASSERT_TRUE(bn::udp_endpoint_open(&sender, 0u, 500u));
        ASSERT_NE(receiver.local_port, 0u);
        ASSERT_NE(sender.local_port, 0u);
        ASSERT_TRUE(bn::udp_endpoint_set_nonblocking(&receiver, true));
    }

    void TearDown() override {
        bn::udp_endpoint_close(&sender);
        bn::udp_endpoint_close(&receiver);
    }

    void send(const void* data, std::uint32_t bytes) {
        ASSERT_EQ(bn::udp_send_to(&sender,
            {bn::k_udp_loopback_ipv4, receiver.local_port, 0u}, data, bytes), bytes);
    }

    bn::UdpReceiveResult receive(void* data, std::uint32_t cap, bn::UdpAddr* peer) {
        assert(data != nullptr);
        assert(cap > 0u);
        bn::UdpReceiveResult result{bn::UdpReceiveStatus::Idle, 0u, 0};
        for (unsigned i = 0; i < kReceiveAttempts; ++i) {
            result = bn::udp_recv_from_checked(&receiver, data, cap, peer);
            if (result.status != bn::UdpReceiveStatus::Idle) return result;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return result;
    }

    std::int64_t receive_legacy(void* data, std::uint32_t cap, bn::UdpAddr* peer) {
        assert(data != nullptr);
        assert(cap > 0u);
        for (unsigned i = 0; i < kReceiveAttempts; ++i) {
            const auto bytes = bn::udp_recv_from(&receiver, data, cap, peer);
            if (bytes >= 0) return bytes;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return -1;
    }

    void expect_peer_unchanged(const bn::UdpAddr& peer) {
        EXPECT_EQ(peer.ipv4, kPeerSentinel.ipv4);
        EXPECT_EQ(peer.port, kPeerSentinel.port);
        EXPECT_EQ(peer._pad, kPeerSentinel._pad);
    }
};
}  // namespace

TEST_F(BoltUdpChecked, ExactCapacityIsCompleteWithPayloadAndPeer) {
    const std::uint8_t sent[]{0x12u, 0x34u, 0x56u, 0x78u};
    send(sent, sizeof(sent));
    std::uint8_t data[sizeof(sent)]{};
    auto peer = kPeerSentinel;
    const auto result = receive(data, sizeof(data), &peer);
    EXPECT_EQ(result.status, bn::UdpReceiveStatus::Complete);
    EXPECT_EQ(result.bytes, sizeof(sent));
    EXPECT_EQ(result.native_error, 0);
    EXPECT_EQ(std::memcmp(data, sent, sizeof(sent)), 0);
    EXPECT_EQ(peer.ipv4, bn::k_udp_loopback_ipv4);
    EXPECT_EQ(peer.port, sender.local_port);
    EXPECT_EQ(peer._pad, 0u);
}

TEST_F(BoltUdpChecked, OversizedPacketIsTruncatedAndNextPacketIsWhole) {
    const std::uint8_t oversized[]{1u, 2u, 3u, 4u, 5u};
    send(oversized, sizeof(oversized));
    std::uint8_t data[4]{};
    auto peer = kPeerSentinel;
    const auto truncated = receive(data, sizeof(data), &peer);
    EXPECT_EQ(truncated.status, bn::UdpReceiveStatus::Truncated);
    EXPECT_EQ(truncated.bytes, 0u);
#if defined(_WIN32)
    EXPECT_NE(truncated.native_error, 0); // recvfrom preserves WSAEMSGSIZE.
#else
    EXPECT_EQ(truncated.native_error, 0); // recvmsg reports output MSG_TRUNC.
#endif
    expect_peer_unchanged(peer);
    const auto empty = bn::udp_recv_from_checked(&receiver, data, sizeof(data), &peer);
    EXPECT_EQ(empty.status, bn::UdpReceiveStatus::Idle);
    EXPECT_EQ(empty.bytes, 0u);
    expect_peer_unchanged(peer);

    const std::uint8_t next[]{9u, 8u, 7u};
    send(next, sizeof(next));
    const auto complete = receive(data, sizeof(data), &peer);
    EXPECT_EQ(complete.status, bn::UdpReceiveStatus::Complete);
    EXPECT_EQ(complete.bytes, sizeof(next));
    EXPECT_EQ(complete.native_error, 0);
    EXPECT_EQ(std::memcmp(data, next, sizeof(next)), 0);
    EXPECT_EQ(peer.ipv4, bn::k_udp_loopback_ipv4);
    EXPECT_EQ(peer.port, sender.local_port);
}

TEST_F(BoltUdpChecked, ZeroLengthDatagramIsCompleteThenIdle) {
    const std::uint8_t unused = 0u;
    send(&unused, 0u);
    std::uint8_t data = 0xAAu;
    auto peer = kPeerSentinel;
    const auto complete = receive(&data, sizeof(data), &peer);
    EXPECT_EQ(complete.status, bn::UdpReceiveStatus::Complete);
    EXPECT_EQ(complete.bytes, 0u);
    EXPECT_EQ(complete.native_error, 0);
    EXPECT_EQ(data, 0xAAu);
    EXPECT_EQ(peer.ipv4, bn::k_udp_loopback_ipv4);
    EXPECT_EQ(peer.port, sender.local_port);

    peer = kPeerSentinel;
    const auto empty = bn::udp_recv_from_checked(&receiver, &data, sizeof(data), &peer);
    EXPECT_EQ(empty.status, bn::UdpReceiveStatus::Idle);
    EXPECT_EQ(empty.bytes, 0u);
    EXPECT_NE(empty.native_error, 0);
    expect_peer_unchanged(peer);
}

TEST_F(BoltUdpChecked, EmptyNonblockingReceiveLeavesPeerUnchanged) {
    std::uint8_t data[4]{};
    auto peer = kPeerSentinel;
    const auto result = bn::udp_recv_from_checked(&receiver, data, sizeof(data), &peer);
    EXPECT_EQ(result.status, bn::UdpReceiveStatus::Idle);
    EXPECT_EQ(result.bytes, 0u);
    EXPECT_NE(result.native_error, 0);
    expect_peer_unchanged(peer);
}

TEST_F(BoltUdpChecked, InvalidInputLeavesPeerAndQueuedPacketUnchanged) {
    const std::uint32_t sent = 0x12345678u;
    send(&sent, sizeof(sent));
    std::uint32_t data = 0u;
    auto peer = kPeerSentinel;
    const auto null_buffer = bn::udp_recv_from_checked(&receiver, nullptr, 4u, &peer);
    const auto zero_cap = bn::udp_recv_from_checked(&receiver, &data, 0u, &peer);
    const auto huge_cap = bn::udp_recv_from_checked(&receiver, &data,
        static_cast<std::uint32_t>(INT_MAX) + 1u, &peer);
    for (const auto result : {null_buffer, zero_cap, huge_cap}) {
        EXPECT_EQ(result.status, bn::UdpReceiveStatus::Error);
        EXPECT_EQ(result.bytes, 0u);
        EXPECT_NE(result.native_error, 0);
    }
    expect_peer_unchanged(peer);
    const auto complete = receive(&data, sizeof(data), &peer);
    EXPECT_EQ(complete.status, bn::UdpReceiveStatus::Complete);
    EXPECT_EQ(complete.bytes, sizeof(sent));
    EXPECT_EQ(data, sent);
}

TEST_F(BoltUdpChecked, ClosedAndNullEndpointsReturnErrorWithoutWritingPeer) {
    bn::udp_endpoint_close(&receiver);
    std::uint32_t data = 0u;
    auto peer = kPeerSentinel;
    const auto closed = bn::udp_recv_from_checked(&receiver, &data, sizeof(data), &peer);
    const auto null_ep = bn::udp_recv_from_checked(nullptr, &data, sizeof(data), &peer);
    const bn::UdpEndpoint invalid{bn::k_udp_invalid_sock - 1u, 0u, {}};
    const auto invalid_handle = bn::udp_recv_from_checked(&invalid, &data, sizeof(data), &peer);
    for (const auto result : {closed, null_ep, invalid_handle}) {
        EXPECT_EQ(result.status, bn::UdpReceiveStatus::Error);
        EXPECT_EQ(result.bytes, 0u);
        EXPECT_NE(result.native_error, 0);
    }
    expect_peer_unchanged(peer);
}

TEST_F(BoltUdpChecked, LegacyReceiveRemainsUsableAlongsideCheckedReceive) {
    const std::uint32_t sent = 12345u;
    send(&sent, sizeof(sent));
    std::uint32_t data = 0u;
    const auto checked = receive(&data, sizeof(data), nullptr);
    EXPECT_EQ(checked.status, bn::UdpReceiveStatus::Complete);
    EXPECT_EQ(checked.bytes, sizeof(sent));
    EXPECT_EQ(data, sent);

    send(&sent, sizeof(sent));
    auto peer = kPeerSentinel;
    EXPECT_EQ(receive_legacy(&data, sizeof(data), &peer), sizeof(sent));
    EXPECT_EQ(data, sent);
    EXPECT_EQ(peer.ipv4, bn::k_udp_loopback_ipv4);
    EXPECT_EQ(peer.port, sender.local_port);
    send(&sent, 0u);
    EXPECT_EQ(receive_legacy(&data, sizeof(data), nullptr), 0);
}
