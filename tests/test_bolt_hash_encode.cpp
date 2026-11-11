// Spark hash/digest/encode/AES kernels (bolt_hash_spark.h, bolt_encode_spark.h, bolt_aes.h).
// Expected digests come from Python hashlib / zlib / cryptography.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "bolt/kernels/bolt_aes.h"
#include "bolt/kernels/bolt_encode_spark.h"
#include "bolt/kernels/bolt_hash_spark.h"

namespace hx = bolt::kernels::hashx;
namespace ex = bolt::kernels::encx;
namespace ae = bolt::kernels::aes;

namespace {

std::string hex(const std::uint8_t* p, std::size_t n) {
    std::string s(2 * n, '\0');
    if (n != 0) ex::hex_upper(p, n, s.data());
    for (char& c : s) c = static_cast<char>(c >= 'A' && c <= 'F' ? c + 32 : c);
    return s;
}

std::string pat(std::size_t n) {
    std::string s(n, '\0');
    for (std::size_t i = 0; i < n; ++i) s[i] = static_cast<char>((i * 7 + 3) % 251);
    return s;
}

const std::uint8_t* u8(const std::string& s) { return reinterpret_cast<const std::uint8_t*>(s.data()); }

struct ShaRow { int len; const char *s224, *s384, *s512, *s256; std::uint32_t crc; };
const ShaRow kSha[] = {
    {0, "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f", "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b", "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0u},
    {1, "15ebd4b8fdfc6905ea002a3efcc8faa71954739ee62bbbf16836eb5c", "8a6c69af6fb6247635f837958446fb8f10e39bd5fbc244f7e635176339a3be614f6394247f01dbe1126c178c7bd48cb5", "e45bf5817ddf94aa2f7a407071f0eedc6beb98f768b4cd33d1176d44d1563a45a5d7212290eb7670c6786b13591aedac86478993895e8b24e612014abaa6ba04", "084fed08b978af4d7d196a7446a86b58009e636b611db16211b65a9aadff29c5", 1259060791u},
    {3, "956a7e9eeefc4f6175e8cd337cd28a80f6a0a988b3fe0ed647b4a953", "de5060203c58b419f136c17666d339a05705575827e3600f0475f6fb3089870aede26c5a0903dea5938bcded4ee9e619", "8853167ab14c043919981a6409307ef68b5b726f4b80dcab259daa24b7cabcb8b4bbfd3e2e36ed98465059661f672e6529cdcc673e7eabfa78dd3186745b112a", "6ab0dba1f4f1dfbb37b4f9eeb092c09fca4900ad32bdcd147d8dde35d6c87c35", 1834528563u},
    {55, "55608cc8d19643d988227c18bd0ae024aa5a666d873e63878892f2ec", "5857bfb2523e03c703868c1dc5ad6c9c29aa20ea9f868eb35079ee9cfc02e870e68dd05e0a7a9c7e6a5318a7f6eb7387", "eac5ae9c6b4ff21e87bd0e3ab17784a3916ea556c4b893b0c24843068cea295d15ce84fe40b14e1c3610c12b08a82b6e5cf35a2a90f76bfb9baa2ecd6f299dbb", "1deace58c745f3ecadde68a5923f494c3703fa73f0306483ccb898a5826e8d70", 1737084725u},
    {56, "79d90db18d878645ce397011141cd599cab051d40a8e4e115d26cb75", "ba012326f3e711530691e2964e229e05f6c52ce5a08da925c684c9e79b9fee8bf2ad74240c6d3dbc4afd5ebf4aef6625", "8b8c39aff78464f48125c952d59a17335db274120c321928993280e3de8ed56ba428d66b772bd2c91d4f6ba49b59d26e4c2cfb08890117aab127ab87afbcd73f", "06dbe23685750e4d3881ded95047abaf93fa8f9c5d3501dc57c717a72ff1398e", 280140261u},
    {57, "44a6541b592e1e1f9cfa3eba32c7bca9e22d1bd2404c8dd5d7cc6b2c", "a661be80dfc9019e705945adb8dec19ebf35c502f9e1e962797fdb2468c5a44cfeeb96484aa0a38b70d55e6713e362ad", "80a8a8349885ce18435b0d0132adc6eb4c5d0f8116be6608e954ce9cca259a8c04c83198c44f7504dd96d01e250e7580a95e1c6f7b8d3eeac9b54caf328091a5", "011f4bced20a249afb834d24b3e4eca31ad5ea2bdd1d8a1a564c2d2410398e69", 4068333735u},
    {63, "0b6975d77e29540cf227b428ff9283bec172270ec3a7cceb74ba7be4", "914e733874699bca094e19b81abc651bd5efd2009aac12de1f2880ed8d4f37eb588cfb44317d615511d8a161f4babaa4", "1955e5263d977825e571a3148f29d53d2d9a305779fdd8de21553a2359e3a149e16510d1b504eaf2b60f2283bfc2897b6346cc70749e097376aa124c958f4d32", "47fb38b12335c9298d09280515c0666489a189d1554bb0ac1a0740806ce9d8b6", 60112578u},
    {64, "66071624305ed69699a64da4209c5990ea95ccaab19ec40b2981250f", "3fc24c04f5cc0d9931ff7b0f4e51693cd84132f3cac401afde2fc48af180b4640c801510615a533db7b5ace8d816963f", "1cf8de2670546a69ca5cd3e93399d4a8e35a1fb79d60d10e80239b902cdc1ff9554a14bca10a6957747585a56c00127f43e4ad8769ca14be1816420a955f5c13", "dfa798724b1a8014994f363e5da7474ed26ce3757fb29e07aa47ad5a9352d37b", 1258826505u},
    {65, "97cbe8ea781b3fef06d36878d0eeb5d6e262638500e8d92c8ef291ef", "ea5da87c44f8b6f5a550efa321877332cb8a6d4200169d34c021e39b67d16257cfd94010ff3d95db6febcf16090802d3", "a5f44d94d65acb5c91488425aea78fea461758a15fbe5f291c0c2d0078a016aac424cddfd725566db48dd5a769ab604f95863c5c27e4e87f1598c998a69f83b0", "dd2eab5a5507d7717c63ce953d9ac61752c21e664425ad1227054722a0d97f69", 1042945408u},
    {111, "63470d4382b4147a88ad2a951716f12b5e2f9947bd38e13fa92ad65d", "15d0faf0d34231c5a3daa35321315f6a9c8185fc9e24240acbb508b5049002b826906bd03b65f4c23ddbf0bd87848e82", "0ef0e383d1e3b7f1e7477f27d7d62a25b3d637c42b509c96f230889cc05cade87cb595834c2f3cccd4c842221251636e3ef6c8b57ee0656e5352a9e968044ab7", "2ea239204423b890039a115a62d44a1b1c37675a784f5c6a9b786e9fca4c4cee", 1828910019u},
    {112, "cdac531ed05ea7a07cf6f4faa10458a776df561a7942e4e34ff5ad64", "3f2f40f7d21ee0ece037ccb1c2ae05f1d9999888c7ef45c7bd731b218422cd7e0b970dd827979c2a35459a030e749ebe", "3b30650ae40d6cf145136a38623ebf064e642157296df6595c6b1a3fac9f5c30461bbc698b19fff7192763656ffc619002ce0cf39eb9efdfd9914a81d81fd71f", "ab7909f0b44839b4f374c4e5974bc1dd7c50eae999602865138f9e3586da29c2", 1516746648u},
    {113, "c026984f0f035652b829be376bb296333dc13f3a34d2d70eca98f196", "9dcd84d6dcaf9e88aab6254734e666115d43e6774a4658ae18fa6d6e91fc0639f6d906a3361e674249512ef49b9cb46c", "56e7afeae695f146fa3ed396eb33c4c0c3e7ff88974c9df24b40327d92889628508cff2c15b8f327e14c739c2d0e7e9670424a0e600b0e396aa789ca7fb734c3", "a6e4d45701eb914359bd05a6d4e54019862393282a6c220158faabc217a327bc", 4193047208u},
    {119, "5bc533f7af6a898a8219febe0eced7adbb44377b6aaf61876b905cf7", "55f1a337b57d89fb9a07d244dea731272025ab4d63346cd6f176aa7a2221c8da17368ba240b2697ccf4a106ee910fd96", "62c4c75e1b2d0edd08d72113ffad4f440a92d5d29d5ad34c7d551bdd38fb8cdf1808202331d7a1e1e2bae5a4378e9818ded76a3ceb15beac1ee8d49e659a99e8", "c6e0f435df5d7d265baacca31e0602c00aa22fa6d3819aed664649294c743756", 2716596935u},
    {120, "386f9a652e1fe4cf55a4b40b965b2e470b923450420e38d3a343bb5d", "b4c0322db078db7cf079dd70e26ede1406c4f96b390c2c95e4c45c99fd0caa6760e8f79a3f43ac414ae80e44724722aa", "7ba1491ca9a9335da6e5000dce692cf69815ef3bcfe2f0e2b949c002216f9a5a4c6fca73abce2c7764f1be9f059b320020118d206ece7e378780d1114c61418c", "17eb8960823a644bde3065620bb9d45931fe8993fd8eb692a17aff0fd725db6a", 633426730u},
    {127, "84d8568700efa0421506b928e07c34dcd87b7348e955672711e5e19c", "7ccfb0aaa18666e62745e4712b5cfa942410522f79d0bd8dd06b73e3d50d676149a9d3160c65b948d29f4444e3577627", "07d05c6a64f1c2ebc735334f428871fca7153286d369318b52427074c3d97e97af523a00b2dd02e5150ee7f962380cf841bcb566cd4921e0226602986e59aff6", "2409a159329f4e770a7d8128130497abb1de01ffb6ad389d97d89ebbeeb7f02c", 2433792045u},
    {128, "35a65dc736672442e303c187794272043c29cc1cd361fa0833fddb3b", "cc793001513820ab7cb27b05590881ca83d7130063454ff76e61b271f5a85818ec751b3b2af1ac3c7f48f2ef86e1ccbe", "9ac7d9bb8e42c2ce0230fe54f60868dca869afe8c4c02771ae54a7827818e6a8840b1fc2efbf2b5163150ed3710318d750c0ac2998b2c628ce4e66cb12cabfff", "3a9c1e22e20578eebe442238f431befd688bd56698eb7c77d95d3f97d8f58207", 3978754536u},
    {129, "9c670a61366d12dea9263f4a30287865183846927fb1f31301bec7c8", "c9f0d67d9f66dbd7e14cd7432f200341f89a7f5d475041d3321ba388eabe798a510dede6101236fb1e934b0177b471a7", "8e14a5884c493da00130dddf9c0140009e7a4fbfebe9c88416bfae10fee38409fa50288087ff2d50e631b0695c95d6bb33a781069487fde48b2fde8d0c4230ab", "956f5da998f608be52587f28902b608909b661f99a1b300e3d3046b4f69f6fb5", 1648316758u},
    {200, "2001c390f3db17de9e4edd7ca5a5efce277fd597d4e25cb8245f318a", "68a55be659a40b4292c66c33d574304f20a17140f063e2d4f330fbbea34f40a770cf4e18b6038212d2fb927131528ae7", "fe955369c891d548bdfb2b84cf6866f31ccc25d7f959cf45fcf6b18f5430a957943124550288b80571af97eb9e2c234d34e92f74867927ec60164db7a6b3806d", "58571ac1c94ce02ff78ca2556530759ff4b9d60ba9d41bee6bb65d0edea9f1c6", 4215937212u},
    {255, "99abcbcd77775b05418947d5a1fab51d96b6b7251c823a90dc8df081", "0c435f9e209e087c592438168c55f4a94147273037f2d4d5cec2fd9e90bc125070c955f18d635969c283910166e8fe1d", "6b4bc2e04c7515a5c49d7e4760ec83eebfcc561eaff834e8966c7485c1227cfa248ffd7e10bf0a6e4b0f7d2920496d7eeae8534ca01ffa7825b5f16591cc734a", "25708de2a70cc8f1562d45807a420ba115cc2e05c9cba175dee089af80f65a29", 3566774305u},
    {256, "2af99a9da616cdbbee33c4d296331d99e398b0f334ff60e053474c46", "03ee5ff18686a7c6ab4b2d4725e0e16a552a9425922a0a6eed7970c444fbe12bbdb9ee78b8302352e7eec6792e5cdc4c", "13b7b7634df11324dc0cba338094805ca25488452ddf08325c14b4f57073d3b4df5ac4088c2ceae9c4b8f8d590704437880f3709f4cadf360b7e9d311c3d8635", "5b1eaad64c2d253d58ed69ccef0eceaf5175bf9ecb153a21b72c429f6fb788e8", 330787470u},
    {1000, "36042fba56241d7289adf173045112503d8a15c886dcd5bd333b91d3", "a02b7ed5eb01c7daa257bb1d8455d1b3528ac3f10c1c949abcb5580e4dfa10d5cdb384cd1eb55fc82a8ce87e8b3e9044", "585b7a26bf81872d575ee2d071374956d4fa3570e31e244231d14410f824a73a64ac870479a54ab425473726b529392e274683daa521d9b89d8b2bf80c0c831f", "a9425c416f534025a4e2422bd14adba4ec3d4a68d10c3329be8df612964d2b6e", 2734237539u},
};

struct AesRow { int klen, len; const char *ecb, *cbc, *gcm; };
const AesRow kAes[] = {
    {16, 0, "954f64f2e4e86e9eee82d20216684899", "a82eb5a05312bb3bd0654452081202cd", "1c925f44898ce0668bd234fff0e842ed"},
    {16, 1, "1de7740872ce6600fe65a9c1f68d3c11", "b8d17f77204631b1e7e81855e7ce0349", "35c658f1b31d09a22539948e9ebd3a6b52"},
    {16, 15, "4af5c4804c880e865e756389bc4e9373", "ec7edb95ffcf7172859c5ffcdfd39a68", "352285a61115061de5d5e112963f06e3937e1d6f77e0f460b92f1f0711e8e0"},
    {16, 16, "268590873ef180ab539cf461a5c5f892954f64f2e4e86e9eee82d20216684899", "f92326b52c4b767d202ce2d6008b083feec76409682d8d30093a6e7efa67ef81", "352285a61115061de5d5e112963f06caec25c2d9c4c91c6cf4081eae84a9a742"},
    {16, 17, "268590873ef180ab539cf461a5c5f892ed10058543320fa617862756ee1f45ba", "f92326b52c4b767d202ce2d6008b083f6fe560e7c23ce2433970956b85c195db", "352285a61115061de5d5e112963f06ca035e0b6d162c2fbc52bf64e1413bfe2f69"},
    {16, 33, "268590873ef180ab539cf461a5c5f8929995d4a8b9338747669ac484fc5403efceac9452c34c1c7d2dc2bf8ab87025c1", "f92326b52c4b767d202ce2d6008b083f4dfabf994bcf84cb0989d3031e2afb02231a59c201ddca0b033b7644306fe6a3", "352285a61115061de5d5e112963f06ca036a7c815eb980b77f05fe08d13124191d6edb0056b1c40fd5ffcbb41942583a5f"},
    {24, 0, "3fe7286abde5f03943d5777020259626", "4a7d26174e981c8e2bc64165a249a513", "6beb7d7eeba204fca3aa8fc7781067a1"},
    {24, 1, "e3380fdeb5021bb9880468e48fd2c4e8", "8f676ddebdaf8658f9cd6aaab4e4c6e1", "ba56a0002b80551cddbf982f6d13bb2535"},
    {24, 15, "be99cc815284fbc7d829eb1076244e22", "efa49a4272ccf9a9f8d020402a64670e", "ba8c0d3089307d80276e5fa7128de77f1355f9c937b1eefcb6b963e41ae835"},
    {24, 16, "3ff9b267a6d17da30ec2fdf1869e402a3fe7286abde5f03943d5777020259626", "36680a41c2c3d94ad1643b8d0d6ceb5837da557ccfdf74324c85d6e6a6eecc4d", "ba8c0d3089307d80276e5fa7128de7da3997613baa7f9e1c71a5d8d7e19c12b0"},
    {24, 17, "3ff9b267a6d17da30ec2fdf1869e402a36d09c32a5eb6abf204540e1f14721a2", "36680a41c2c3d94ad1643b8d0d6ceb58f34e889f3b848b19b8920c963a62911a", "ba8c0d3089307d80276e5fa7128de7da441328eaccd5f864ca95378abb6410474a"},
    {24, 33, "3ff9b267a6d17da30ec2fdf1869e402ab71942f6403221e0e862eb43f02b19aaf749a64bb8230396bd2da24e4b0aa422", "36680a41c2c3d94ad1643b8d0d6ceb58945c77832b5fc12314b6dda16221fe442a13078f0118e2d67ecd9c49f06fcf62", "ba8c0d3089307d80276e5fa7128de7da4482e370ccd78a1d389d92d60194803add096814aa5e9bf3d5621c1fa523b2eb62"},
    {32, 0, "9f3b7504926f8bd36e3118e903a4cd4a", "f795f6e8802fd151a021a41150d4ab69", "325d2a68a2886a11edbe04890bf68339"},
    {32, 1, "940afa3663feacb1ae12bb5cf805f9d0", "d8dea5da5d93cf441e60fd5704f8c1f2", "123514bff7a44216c8fc35ef49cee60def"},
    {32, 15, "68b890ac57333f312253279b330a7d10", "ce4928e54a56fd9826b1b50952000d51", "127270c399ead6144f7e56ee49588eb88147f6c7d21207b27e070468bd341a"},
    {32, 16, "ae5bf9ddaad53f23deb4efae9db724629f3b7504926f8bd36e3118e903a4cd4a", "c520fa7e867b82c5b6c01f57ec10a9e35527c50b8d975f22490d501d5b5428bd", "127270c399ead6144f7e56ee49588e78f77cc6ac853d331fbb9e1cd69f377573"},
    {32, 17, "ae5bf9ddaad53f23deb4efae9db72462bae2104eae8b167a58122a0ed4b0f1a0", "c520fa7e867b82c5b6c01f57ec10a9e3c41c113d106c3ae53d4c130d9f4049f0", "127270c399ead6144f7e56ee49588e78936a386ee30ac2aeb37a84a0eadae7f44c"},
    {32, 33, "ae5bf9ddaad53f23deb4efae9db72462c6e8b0d060f9d45d8e1247da4ab53f50105d9440a34b1901d64b5dcf087b95f0", "c520fa7e867b82c5b6c01f57ec10a9e3f9c6174e35362b762936c7702d95bd241f85adb336ec27ca918280c8303b938f", "127270c399ead6144f7e56ee49588e7893fd6d208bfd3580461b82211d5f7e352fa9421b983acaf28127304e48630fdcdf"},
};

std::string key32() { std::string k(32, '\0'); for (int i = 0; i < 32; ++i) k[static_cast<std::size_t>(i)] = static_cast<char>(i); return k; }
std::string ivn(int base, int n) { std::string k(static_cast<std::size_t>(n), '\0'); for (int i = 0; i < n; ++i) k[static_cast<std::size_t>(i)] = static_cast<char>(base + i); return k; }

}  // namespace

TEST(HashSpark, Sha2AndCrc32AgainstHashlib) {
    for (const ShaRow& r : kSha) {
        const std::string d = pat(static_cast<std::size_t>(r.len));
        std::uint8_t out[64];
        hx::sha2(u8(d), d.size(), 224, out);
        EXPECT_EQ(hex(out, 28), r.s224) << r.len;
        hx::sha2(u8(d), d.size(), 256, out);
        EXPECT_EQ(hex(out, 32), r.s256) << r.len;
        hx::sha2(u8(d), d.size(), 384, out);
        EXPECT_EQ(hex(out, 48), r.s384) << r.len;
        hx::sha2(u8(d), d.size(), 512, out);
        EXPECT_EQ(hex(out, 64), r.s512) << r.len;
        EXPECT_EQ(hx::crc32(u8(d), d.size()), r.crc) << r.len;
    }
    EXPECT_EQ(hx::sha2_digest_bytes(0), 32u);
    EXPECT_EQ(hx::sha2_digest_bytes(1), 0u);
    EXPECT_EQ(hx::sha2_digest_bytes(225), 0u);
}

TEST(HashSpark, MurmurAndXxhashStructure) {
    // The tail of hashUnsafeBytes is one round per byte: "ab" must not equal stock murmur3 of "ab".
    EXPECT_NE(hx::m3_hash_bytes(u8(std::string("ab")), 2, 42), hx::m3_hash_int(0x6261u, 42));
    // A 4-byte value hashed as int, as long (sign-extended) and as bytes are three different rounds.
    EXPECT_EQ(hx::m3_hash_long(static_cast<std::uint64_t>(-1), 42), hx::m3_hash_long(0xFFFFFFFFFFFFFFFFull, 42));
    // Determinism and seed chaining.
    EXPECT_EQ(hx::xx_hash_bytes(nullptr, 0, 42), hx::xx_hash_bytes(nullptr, 0, 42));
    EXPECT_NE(hx::xx_hash_bytes(nullptr, 0, 42), hx::xx_hash_bytes(nullptr, 0, 43));
    // XXH64 of the empty input with seed 0 is the published constant.
    EXPECT_EQ(hx::xx_hash_bytes(nullptr, 0, 0), 0xEF46DB3751D8E999ull);
    // XXH64("abc", seed 0) published value.
    EXPECT_EQ(hx::xx_hash_bytes(u8(std::string("abc")), 3, 0), 0x44BC2CF5AD770999ull);
}

TEST(EncodeSpark, HexBinBase64) {
    char b[80];
    EXPECT_EQ(std::string(b, ex::hex_long(-1, b)), "FFFFFFFFFFFFFFFF");
    EXPECT_EQ(std::string(b, ex::hex_long(0, b)), "0");
    EXPECT_EQ(std::string(b, ex::bin_long(13, b)), "1101");
    EXPECT_EQ(ex::bin_long(-1, b), 64u);
    std::uint8_t d[8];
    std::uint64_t n = 0;
    EXPECT_TRUE(ex::unhex(u8(std::string("414")), 3, d, &n));
    EXPECT_EQ(hex(d, n), "0414");
    EXPECT_FALSE(ex::unhex(u8(std::string("4G")), 2, d, &n));
    const std::string in = pat(100);
    char enc[256];
    const std::uint64_t el = ex::base64_encode(u8(in), in.size(), true, enc);
    EXPECT_NE(std::string(enc, el).find("\r\n"), std::string::npos);
    std::uint8_t dec[256];
    std::uint64_t dl = 0;
    EXPECT_EQ(ex::base64_decode(reinterpret_cast<const std::uint8_t*>(enc), el, true, dec, &dl), ex::B64Status::Ok);
    EXPECT_EQ(std::string(reinterpret_cast<char*>(dec), dl), in);
    EXPECT_EQ(ex::base64_decode(reinterpret_cast<const std::uint8_t*>(enc), el, false, dec, &dl), ex::B64Status::IllegalChar);
    EXPECT_EQ(ex::base64_decode(u8(std::string("A")), 1, true, dec, &dl), ex::B64Status::LastUnitShort);
    EXPECT_EQ(ex::base64_decode(u8(std::string("AAH/==")), 6, true, dec, &dl), ex::B64Status::WrongEndingUnit);
}

TEST(EncodeSpark, Conv) {
    char o[70];
    std::uint32_t n = 0;
    EXPECT_EQ(ex::conv(u8(std::string("-10")), 3, 16, 10, true, o, &n), ex::ConvStatus::Ok);
    EXPECT_EQ(std::string(o, n), "18446744073709551600");
    EXPECT_EQ(ex::conv(u8(std::string("-10")), 3, 16, -10, true, o, &n), ex::ConvStatus::Ok);
    EXPECT_EQ(std::string(o, n), "-16");
    EXPECT_EQ(ex::conv(u8(std::string("18446744073709551616")), 20, 10, 16, true, o, &n), ex::ConvStatus::Overflow);
    EXPECT_EQ(ex::conv(u8(std::string("18446744073709551616")), 20, 10, 16, false, o, &n), ex::ConvStatus::Ok);
    EXPECT_EQ(std::string(o, n), "FFFFFFFFFFFFFFFF");
    EXPECT_EQ(ex::conv(u8(std::string("1a2")), 3, 10, 2, true, o, &n), ex::ConvStatus::Ok);
    EXPECT_EQ(std::string(o, n), "1");
}

TEST(EncodeSpark, Charsets) {
    std::uint8_t o[64];
    std::uint64_t n = 0;
    const std::string s = "a\xC3\xA9";   // "a" + e acute
    EXPECT_EQ(ex::charset_encode(u8(s), s.size(), ex::Charset::Utf16, o, &n), ex::CodingStatus::Ok);
    EXPECT_EQ(hex(o, n), "feff006100e9");
    EXPECT_EQ(ex::charset_encode(u8(s), s.size(), ex::Charset::UsAscii, o, &n), ex::CodingStatus::Malformed);
    EXPECT_EQ(ex::charset_encode(u8(s), s.size(), ex::Charset::Latin1, o, &n), ex::CodingStatus::Ok);
    EXPECT_EQ(hex(o, n), "61e9");
    const std::string bad = "\x61\xFF";
    EXPECT_EQ(ex::charset_decode(u8(bad), bad.size(), ex::Charset::Utf8, o, &n), ex::CodingStatus::Malformed);
    const std::string le("\xFF\xFE\x61\x00", 4);
    EXPECT_EQ(ex::charset_decode(u8(le), le.size(), ex::Charset::Utf16, o, &n), ex::CodingStatus::Ok);
    EXPECT_EQ(hex(o, n), "61");
    EXPECT_EQ(ex::charset_of("UTF-8", 5), ex::Charset::Utf8);
    EXPECT_EQ(ex::charset_of("Utf8", 4), ex::Charset::None);
}

TEST(AesSpark, EcbCbcGcmAgainstCryptography) {
    const std::string key = key32();
    const std::string iv16 = ivn(100, 16), iv12 = ivn(200, 12);
    for (const AesRow& r : kAes) {
        const std::string d = pat(static_cast<std::size_t>(r.len));
        ae::AesKey k;
        ASSERT_TRUE(ae::key_init(u8(key), static_cast<std::uint32_t>(r.klen), &k));
        std::uint8_t out[128];
        ae::encrypt_padded(k, nullptr, u8(d), d.size(), out);
        EXPECT_EQ(hex(out, ae::padded_len(d.size())), r.ecb) << r.klen << "/" << r.len;
        std::uint8_t back[128];
        std::uint64_t bn = 0;
        EXPECT_EQ(ae::decrypt_padded(k, nullptr, out, ae::padded_len(d.size()), back, &bn), ae::AesStatus::Ok);
        EXPECT_EQ(std::string(reinterpret_cast<char*>(back), bn), d);
        ae::encrypt_padded(k, u8(iv16), u8(d), d.size(), out);
        EXPECT_EQ(hex(out, ae::padded_len(d.size())), r.cbc) << r.klen << "/" << r.len;
        EXPECT_EQ(ae::decrypt_padded(k, u8(iv16), out, ae::padded_len(d.size()), back, &bn), ae::AesStatus::Ok);
        EXPECT_EQ(std::string(reinterpret_cast<char*>(back), bn), d);
        ae::gcm_encrypt(k, u8(iv12), u8(std::string("hdr")), 3, u8(d), d.size(), out);
        EXPECT_EQ(hex(out, d.size() + 16), r.gcm) << r.klen << "/" << r.len;
        EXPECT_EQ(ae::gcm_decrypt(k, u8(iv12), u8(std::string("hdr")), 3, out, d.size() + 16, back), ae::AesStatus::Ok);
        out[0] ^= 1;
        EXPECT_NE(ae::gcm_decrypt(k, u8(iv12), u8(std::string("hdr")), 3, out, d.size() + 16, back), ae::AesStatus::Ok);
    }
    ae::AesKey bad;
    EXPECT_FALSE(ae::key_init(u8(key), 15, &bad));
}
