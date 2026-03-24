import pprint

# 符号概率
prob = {
    'A': 7/8,
    'B': 3/64,
    'C': 2/64,
    'D': 1/64,
    'E': 1/64,
    'F': 1/64,
}

# 构建累计概率表
def build_cdf(prob):
    cdf = {}
    low = 0
    for k, v in prob.items():
        high = low + v
        cdf[k] = (low, high)
        low = high
    return cdf

cdf = build_cdf(prob)

print("符号累计区间：")
pprint.pprint(cdf)


# ----------------------
# 算术编码
# ----------------------

def arithmetic_encode(message, cdf):

    low = 0.0
    high = 1.0

    print("\n编码过程：")

    for i, symbol in enumerate(message):

        range_width = high - low

        sym_low, sym_high = cdf[symbol]

        new_low = low + range_width * sym_low
        new_high = low + range_width * sym_high

        print(f"\nStep {i+1}")
        print("symbol:", symbol)
        print("旧区间: [{:.16f}, {:.16f}]".format(low, high))
        print("symbol区间:", (sym_low, sym_high))
        print("新区间: [{:.16f}, {:.16f}]".format(new_low, new_high))

        low, high = new_low, new_high

    code = (low + high) / 2

    print("\n最终编码值:", code)

    return code, len(message)


# ----------------------
# 算术解码
# ----------------------

def arithmetic_decode(code, length, cdf):

    low = 0.0
    high = 1.0
    message = ""

    print("\n解码过程：")

    for i in range(length):

        range_width = high - low
        value = (code - low) / range_width

        print(f"\nStep {i+1}")
        print("当前区间: [{:.6f}, {:.6f}]".format(low, high))
        print("归一化值:", value)

        for symbol, (sym_low, sym_high) in cdf.items():

            if sym_low <= value < sym_high:

                message += symbol

                new_low = low + range_width * sym_low
                new_high = low + range_width * sym_high

                print("解码符号:", symbol)
                print("新区间: [{:.6f}, {:.6f}]".format(new_low, new_high))

                low, high = new_low, new_high
                break

    return message


# ----------------------
# 测试
# ----------------------

message = "BAAAAAAACAAAAAAAFAAAAAAADAAAAAAAEAAAAAAACAAAAAAABAAAAAAABAAAAAAA"

code, length = arithmetic_encode(message, cdf)

decoded = arithmetic_decode(code, length, cdf)

print("\n原始消息:", message)
print("解码结果:", decoded)