
def merge(filename):
    src = open('data/' + filename + '.src')
    trg = open('data/' + filename + '.trg')

    out = open('data/' + filename + '.csv', 'w')
    for s, t in zip(src, trg):
        out.write(s.strip() + '\t' + t.strip() + '\n')
    out.close()

if __name__ == "__main__":
    merge('train')
    merge('val')
    merge('test')
