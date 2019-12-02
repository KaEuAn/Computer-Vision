fi = open("somecode.txt", 'r')
print('hi')
for line in fi.readlines():
    for symbol in line:
        if symbol == '(':
            print('[', end = '')
        elif symbol == ')':
            print(']', end='')
        else:
            print(symbol, end='')