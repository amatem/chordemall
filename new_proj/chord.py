import mir_eval

QUALITIES = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'maj9':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '9':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'min11':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '11':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#11':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj13':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min13':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b13':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '1':       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

chord_map = {}
rev_chord_map = {}
cnt = 1
for qual in ['maj', 'min']:
    for root in ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']:
        chord_map['{}:{}'.format(root,qual)] = cnt
        rev_chord_map[cnt] = '{}:{}'.format(root,qual)
        cnt =cnt+ 1
    chord_map['{}:{}'.format('D#', qual)] = chord_map['{}:{}'.format('Eb', qual)]
    chord_map['{}:{}'.format('Db', qual)] = chord_map['{}:{}'.format('C#', qual)]
    chord_map['{}:{}'.format('Gb', qual)] = chord_map['{}:{}'.format('F#', qual)]
    chord_map['{}:{}'.format('Ab', qual)] = chord_map['{}:{}'.format('G#', qual)]
chord_map['N']=cnt
rev_chord_map[cnt] = 'N'

def chord_to_category(chord):
	return chord_map[chord]

def category_to_chord(cat):
    return rev_chord_map[cat]

def is_subset(bitmap, ground):
    for i in range(len(bitmap)):
        if bitmap[i] == 0 and ground[i] == 1:
            return False
    return True 

def get_majminchord(chord):
    root, quality, bass = mir_eval.chord.encode(chord)
    root_char = mir_eval.chord.split(chord)[0]
    if root_char == 'N':
        return root_char
    rev = 'N'
    if is_subset(quality, QUALITIES['maj']):
       rev = "{}:{}".format(root_char, 'maj')
    if is_subset(quality, QUALITIES['min']):
       rev = "{}:{}".format(root_char, 'min')
    return category_to_chord(chord_to_category(rev))
