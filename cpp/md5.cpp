#include "md5.h"
#include <cstring>
#include <sstream>
#include <iomanip>

MD5::MD5() {
    init();
}

void MD5::update(const unsigned char* data, size_t len) {
    unsigned int index = (unsigned int)((count[0] >> 3) & 0x3F);
    if ((count[0] += (len << 3)) < (len << 3)) {
        count[1]++;
    }
    count[1] += (len >> 29);

    unsigned int partLen = 64 - index;
    unsigned int i = 0;

    if (len >= partLen) {
        memcpy(&buffer[index], data, partLen);
        transform(buffer);
        for (i = partLen; i + 63 < len; i += 64) {
            transform(&data[i]);
        }
        index = 0;
    }

    memcpy(&buffer[index], &data[i], len - i);
}

void MD5::finalize() {
    static unsigned char padding[64] = { 0x80 };
    unsigned char bits[8];
    encode(bits, count, 8);

    unsigned int index = (unsigned int)((count[0] >> 3) & 0x3f);
    unsigned int padLen = (index < 56) ? (56 - index) : (120 - index);
    update(padding, padLen);
    update(bits, 8);

    encode(digest, state, 16);
    memset(buffer, 0, sizeof(buffer));
    memset(count, 0, sizeof(count));
}

std::string MD5::hexdigest() const {
    std::ostringstream oss;
    for (int i = 0; i < 16; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }
    return oss.str();
}

void MD5::init() {
    count[0] = count[1] = 0;
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;
}

void MD5::transform(const unsigned char block[64]) {
    static const unsigned int S[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
    };

    static const unsigned int T[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
    };

    unsigned int a = state[0], b = state[1], c = state[2], d = state[3], x[16];
    decode(x, block, 64);

    for (int i = 0; i < 64; i++) {
        unsigned int f, g;
        if (i < 16) {
            f = (b & c) | (~b & d);
            g = i;
        } else if (i < 32) {
            f = (d & b) | (~d & c);
            g = (5 * i + 1) % 16;
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (3 * i + 5) % 16;
        } else {
            f = c ^ (b | ~d);
            g = (7 * i) % 16;
        }

        unsigned int temp = d;
        d = c;
        c = b;
        b = b + rotate_left((a + f + T[i] + x[g]), S[i]);
        a = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;

    memset(x, 0, sizeof(x));
}

unsigned int MD5::rotate_left(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

void MD5::encode(unsigned char* output, const unsigned int* input, unsigned int len) {
    for (unsigned int i = 0, j = 0; j < len; i++, j += 4) {
        output[j] = (unsigned char)(input[i] & 0xff);
        output[j + 1] = (unsigned char)((input[i] >> 8) & 0xff);
        output[j + 2] = (unsigned char)((input[i] >> 16) & 0xff);
        output[j + 3] = (unsigned char)((input[i] >> 24) & 0xff);
    }
}

void MD5::decode(unsigned int* output, const unsigned char* input, unsigned int len) {
    for (unsigned int i = 0, j = 0; j < len; i++, j += 4) {
        output[i] = ((unsigned int)input[j]) | (((unsigned int)input[j + 1]) << 8) |
                    (((unsigned int)input[j + 2]) << 16) | (((unsigned int)input[j + 3]) << 24);
    }
}

std::string md5(const std::string& input) {
    MD5 md5;
    md5.update((unsigned char*)input.c_str(), input.size());
    md5.finalize();
    return md5.hexdigest();
}