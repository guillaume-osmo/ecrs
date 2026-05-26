#ifndef MD5_H
#define MD5_H

#include <string>

class MD5 {
public:
    MD5();
    void update(const unsigned char* data, size_t len);
    void finalize();
    std::string hexdigest() const;

private:
    void init();
    void transform(const unsigned char block[64]);
    unsigned int rotate_left(unsigned int x, unsigned int n);
    void encode(unsigned char* output, const unsigned int* input, unsigned int len);
    void decode(unsigned int* output, const unsigned char* input, unsigned int len);

    unsigned int state[4];
    unsigned int count[2];
    unsigned char buffer[64];
    unsigned char digest[16];
};

std::string md5(const std::string& input);

#endif // MD5_H