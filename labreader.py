import glob, os
import pprint

def read_lab_file(path):
    rev = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            raw = line.split()
            if len(raw) > 3:
                print("Irregular line found: {}".format(raw))
            rev.append((float(raw[0]), float(raw[1]), raw[2]))

    return rev

def read_all(folder):
    res = {}
    for f in os.listdir(folder):
        if f.endswith('.lab'):
            res[f] = read_lab_file(os.path.join(folder, f))
    return res

def main():
    data = read_all('/Users/neumann/bilkent/eee485/project/chordemall/data/beatles/chordlab/The Beatles/01_-_Please_Please_Me')
    pp = pprint.PrettyPrinter(indent=3)
    alph = {}
    for key in data:
        print('FILENAME: {}'.format(key))
        #pp.pprint(data[key])
        for chord in data[key]:
            if not chord[2] in alph:
                alph[chord[2]] = 1;
            else:
                alph[chord[2]] += 1;
    print("ALPHABETSIZE: {}".format(len(alph)))
    pp.pprint(alph)

if __name__ == '__main__':
    main()
