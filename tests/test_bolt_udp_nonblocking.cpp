#include <gtest/gtest.h>
#include <chrono>
#include <thread>
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
