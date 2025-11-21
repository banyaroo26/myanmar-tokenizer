#!/usr/bin/env python
# -*- coding=utf-8 -*-

import codecs
# Note: python_utils import removed since it's not provided
import sys
import re
import os
import time

# Fix unichr to chr for Python 3
MyanmarCharacterCode = [('\u1000', '\u109F'),  # Myanmar
                        ('\uAA60', '\uAA7F'),  # Myanmar Extended-A
                        ('\uA9E0', '\uA9FF'),  # Myanmar Extended-B
                        ]


class MyanmarTokenizer:
    # Category name
    _CATEGORY_NAMES = ['C', 'M', 'V', 'S', 'A', 'F', 'I', 'E', 'G', 'D', 'P', 'W']
    # Category's Unicode Code Point
    _CATEGORY_RANGE = [
        ['C', range(0x1000, 0x1021 + 1)],  # Consonants
        ['M', range(0x103B, 0x103E + 1)],  # Medials
        ['V', range(0x102B, 0x1032 + 1)],  # Dependent Vowel Signs
        ['S', [0x1039]],  # Myanmar Sign Virama
        ['A', [0x103A]],  # Myanmar Sign Asat
        ['F', range(0x1036, 0x1038 + 1)],  # Dependent Various Signs
        ['I', [0x1024, 0x1027, 0x102A, 0x104C, 0x104D, 0x104F]],  # Independent Vowels,Independent Various Signs
        ['E', [0x1023, 0x1025, 0x1026, 0x1029, 0x104E]],  # Independent Vowels,Myanmar Symbol Aforementioned
        ['G', [0x103F]],  # Myanmar Letter Great Sa
        ['D', range(0x1040, 0x1049 + 1)],  # Myanmar Digits
        ['P', range(0x104A, 0x104B + 1)],  # Punctuation Marks
        ['W', [0x0020]],  # White space
    ]
    # Myanmar codes
    _MYANMAR_CODES_START = 0x1000
    _MYANMAR_CODES_END = 0x109f
    _MYANMAR_CODES = [chr(n) for n in range(_MYANMAR_CODES_START, _MYANMAR_CODES_END + 1)]
    _PATTERN_MYANMAR_CODES = re.compile('([{}-{}]+)'.format(chr(_MYANMAR_CODES_START), chr(_MYANMAR_CODES_END)),
                                        re.U)

    # Syllable Break Status and Definition
    _BREAK_STATUS_UNDEFINED = -2  # undefined cases
    _BREAK_STATUS_ILLEGAL_SPELLING_ORDER = -1  # Illegal spelling order
    _BREAK_STATUS_NO_BREAK_AFTER_1ST_CHARACTER = 0  # No break after 1 st character
    _BREAK_STATUS_BREAK_AFTER_1ST_CHARACTER = 1  # Break after 1 st character
    _BREAK_STATUS_BREAK_AFTER_2ND_CHARACTER = 2  # Break after 2 nd character
    _BREAK_STATUS_BREAK_AFTER_3RD_CHARACTER = 3  # Break after 3 rd character
    _BREAK_STATUS_BREAK_AFTER_4TH_CHARACTER = 4  # Break after 4 th character

    # Letter Sequence Table: 2nd, 3rd, 4th character
    _LETTER_SEQUENCE_TABLE_INDEX = {c: i for i, c in enumerate('A C D E F G I M P S V W'.split())}
    _LETTER_SEQUENCE_TABLE_2ND_CHARACTER = {
        # A C D E F G I M P S V W
        'A': (-1, -2, 1, 1, 0, -1, 1, 0, 1, 0, 0, 1),
        'C': (0, -2, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1),
        'D': (-1, 1, 0, 1, -1, -1, 1, -1, 1, -1, -1, 1),
        'E': (-1, -2, 1, 1, 2, 0, 1, -1, 1, -1, 0, 1),
        'F': (-1, -2, 1, 1, 2, -1, 1, -1, 1, -1, -1, 1),
        'G': (-1, 1, 1, 1, 0, -1, 1, -1, 1, -1, 0, 1),
        'I': (-1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1),
        'M': (2, -2, 1, 1, 0, 0, 1, 0, 1, -1, 0, 1),
        'P': (-1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1),
        'S': (-1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
        'V': (2, -2, 1, 1, 0, 0, 1, -1, 1, -1, 0, 1),
        'W': (-1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 0),
    }
    _LETTER_SEQUENCE_TABLE_3RD_CHARACTER = {
        # A C D E F G I M P S V W
        'AC': (3, 1, 1, 1, 1, 1, 1, -2, 1, 1, 1, 1),
        'CC': (0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1),
        'EC': (0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1),
        'FC': (3, 1, 1, 1, 1, 1, 1, -2, 1, 1, 1, 1),
        'MC': (0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1),
        'VC': (0, 1, 1, 1, 1, 1, 1, -2, 1, 0, 1, 1),
    }
    _LETTER_SEQUENCE_TABLE_4TH_CHARACTER = {
        # A C D E F G I M P S V W
        'ACM': (4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        'FCM': (4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        'VCM': (4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    }

    @property
    def separator(self):
        return self._separator

    @separator.setter
    def separator(self, value):
        self._separator = value

    def __init__(self, separator='|'):
        self._separator = separator
        for name in MyanmarTokenizer._CATEGORY_NAMES:
            setattr(MyanmarTokenizer, 'Category' + name, name)
        # Each Myanmar character corresponding Category
        self.codeCategory = {chr(_code): _name
                             for _name, _range in MyanmarTokenizer._CATEGORY_RANGE
                             for _code in _range}

    def code2Category(self, sentence):
        # Remove utils.toUnicode call - assume input is already proper string
        return ''.join([self.codeCategory[c] if c in MyanmarTokenizer._MYANMAR_CODES else '?' for c in sentence])

    def _getSyllableBreakStatus(self, categorys, categorysLen):
        if categorysLen == 2:
            letterSequenceTable = MyanmarTokenizer._LETTER_SEQUENCE_TABLE_2ND_CHARACTER
        elif categorysLen == 3:
            letterSequenceTable = MyanmarTokenizer._LETTER_SEQUENCE_TABLE_3RD_CHARACTER
        elif categorysLen == 4:
            letterSequenceTable = MyanmarTokenizer._LETTER_SEQUENCE_TABLE_4TH_CHARACTER
        else:
            letterSequenceTable = None
        if letterSequenceTable is not None:
            status = letterSequenceTable.get(categorys[:categorysLen - 1])
            if status is not None:
                index = MyanmarTokenizer._LETTER_SEQUENCE_TABLE_INDEX.get(categorys[categorysLen - 1])
                if index is not None:
                    return status[index]
        return MyanmarTokenizer._BREAK_STATUS_UNDEFINED

    def cutRecursively(self, sentence):
        # Remove utils.toUnicode call
        result = ''
        for i, s in enumerate(self._split(sentence)):
            if ord(s[0]) < MyanmarTokenizer._MYANMAR_CODES_START \
                    or ord(s[0]) > MyanmarTokenizer._MYANMAR_CODES_END:
                if i != 0: 
                    result += self.separator
                result += s + self.separator
                continue
            categorys = self.code2Category(s)
            result += self._syllableSegmentationRecursively('', categorys, '', s)[1]
        return result

    def _split(self, sentence):
        for s in MyanmarTokenizer._PATTERN_MYANMAR_CODES.split(sentence):
            if s == '':
                continue
            yield s

    def cut(self, sentence):
        # Remove utils.toUnicode call
        result = ''
        for i, s in enumerate(self._split(sentence)):
            if ord(s[0]) < MyanmarTokenizer._MYANMAR_CODES_START \
                    or ord(s[0]) > MyanmarTokenizer._MYANMAR_CODES_END:
                if i != 0: 
                    result += self.separator
                result += s + self.separator
                continue
            categorys = self.code2Category(s)
            result += self._syllableSegmentation(categorys, s)[1]
        return result

    def cutStd(self, stdin, stdout):
        for line in stdin:
            result = self.cut(line.strip())
            stdout.write(result + os.linesep)

    def cutCategory(self, categorys):
        result = self._syllableSegmentationRecursively('', categorys, '', categorys)
        return result[0]

    def _syllableSegmentation(self, categorys, sentence):
        result = ["", ""]
        residueLen = len(categorys)
        sentenceLen = len(categorys)
        start = 0
        while residueLen != 0:
            if residueLen == 1:
                return [result[0] + categorys[-1], result[1] + sentence[-1]]

            breakStatus = self._getSyllableBreakStatus(categorys[start:start + 2], 2)
            if breakStatus == MyanmarTokenizer._BREAK_STATUS_UNDEFINED and residueLen >= 3:
                breakStatus = self._getSyllableBreakStatus(categorys[start:start + 3], 3)
            if breakStatus == MyanmarTokenizer._BREAK_STATUS_UNDEFINED and residueLen >= 4:
                breakStatus = self._getSyllableBreakStatus(categorys[start:start + 4], 4)

            if breakStatus == MyanmarTokenizer._BREAK_STATUS_UNDEFINED:
                categorysLeft = categorys[start] + self.separator
                sentenceLeft = sentence[start] + self.separator
                start += 1
            elif breakStatus == MyanmarTokenizer._BREAK_STATUS_ILLEGAL_SPELLING_ORDER:
                categorysLeft = categorys[start:start + 2] + '?'
                sentenceLeft = sentence[start:start + 2] + '?'
                start += 2
            elif breakStatus == MyanmarTokenizer._BREAK_STATUS_NO_BREAK_AFTER_1ST_CHARACTER:
                categorysLeft = categorys[start]
                sentenceLeft = sentence[start]
                start += 1
            elif breakStatus == MyanmarTokenizer._BREAK_STATUS_BREAK_AFTER_1ST_CHARACTER:
                categorysLeft = categorys[start] + self.separator
                sentenceLeft = sentence[start] + self.separator
                start += 1
            elif breakStatus == MyanmarTokenizer._BREAK_STATUS_BREAK_AFTER_2ND_CHARACTER:
                categorysLeft = categorys[start:start + 2] + self.separator
                sentenceLeft = sentence[start:start + 2] + self.separator
                start += 2
            elif breakStatus == MyanmarTokenizer._BREAK_STATUS_BREAK_AFTER_3RD_CHARACTER:
                categorysLeft = categorys[start:start + 3] + self.separator
                sentenceLeft = sentence[start:start + 3] + self.separator
                start += 3
            elif breakStatus == MyanmarTokenizer._BREAK_STATUS_BREAK_AFTER_4TH_CHARACTER:
                categorysLeft = categorys[start:start + 4] + self.separator
                sentenceLeft = sentence[start:start + 4] + self.separator
                start += 4
            result[0] += categorysLeft
            result[1] += sentenceLeft
            residueLen = sentenceLen - start

        return (result[0].rstrip(self.separator), result[1].rstrip(self.separator))

    def _syllableSegmentationRecursively(self, categorysLeft, categorysRight, sentenceLeft, sentenceRight):
        rightLen = len(categorysRight)
        if rightLen == 0:
            return (categorysLeft, sentenceLeft)
        if rightLen == 1:
            return (categorysLeft + categorysRight, sentenceLeft + sentenceRight)

        breakStatus = self._getSyllableBreakStatus(categorysRight[:2], 2)
        if breakStatus == MyanmarTokenizer._BREAK_STATUS_UNDEFINED and rightLen >= 3:
            breakStatus = self._getSyllableBreakStatus(categorysRight[:3], 3)
        if breakStatus == MyanmarTokenizer._BREAK_STATUS_UNDEFINED and rightLen >= 4:
            breakStatus = self._getSyllableBreakStatus(categorysRight[:4], 4)
            
        if breakStatus == MyanmarTokenizer._BREAK_STATUS_UNDEFINED:
            return self._syllableSegmentationRecursively(categorysLeft + categorysRight[0] + self.separator,
                                                         categorysRight[1:],
                                                         sentenceLeft + sentenceRight[0] + self.separator,
                                                         sentenceRight[1:])
        elif breakStatus == MyanmarTokenizer._BREAK_STATUS_ILLEGAL_SPELLING_ORDER:
            return self._syllableSegmentationRecursively(categorysLeft + categorysRight[:2] + '?', categorysRight[2:],
                                                         sentenceLeft + sentenceRight[:2] + '?', sentenceRight[2:])
        elif breakStatus == MyanmarTokenizer._BREAK_STATUS_NO_BREAK_AFTER_1ST_CHARACTER:
            return self._syllableSegmentationRecursively(categorysLeft + categorysRight[0], categorysRight[1:],
                                                         sentenceLeft + sentenceRight[0], sentenceRight[1:])
        elif breakStatus == MyanmarTokenizer._BREAK_STATUS_BREAK_AFTER_1ST_CHARACTER:
            return self._syllableSegmentationRecursively(categorysLeft + categorysRight[0] + self.separator,
                                                         categorysRight[1:],
                                                         sentenceLeft + sentenceRight[0] + self.separator,
                                                         sentenceRight[1:])
        elif breakStatus == MyanmarTokenizer._BREAK_STATUS_BREAK_AFTER_2ND_CHARACTER:
            return self._syllableSegmentationRecursively(categorysLeft + categorysRight[:2] + self.separator,
                                                         categorysRight[2:],
                                                         sentenceLeft + sentenceRight[:2] + self.separator,
                                                         sentenceRight[2:])
        elif breakStatus == MyanmarTokenizer._BREAK_STATUS_BREAK_AFTER_3RD_CHARACTER:
            return self._syllableSegmentationRecursively(categorysLeft + categorysRight[:3] + self.separator,
                                                         categorysRight[3:],
                                                         sentenceLeft + sentenceRight[:3] + self.separator,
                                                         sentenceRight[3:])
        elif breakStatus == MyanmarTokenizer._BREAK_STATUS_BREAK_AFTER_4TH_CHARACTER:
            return self._syllableSegmentationRecursively(categorysLeft + categorysRight[:4] + self.separator,
                                                         categorysRight[4:],
                                                         sentenceLeft + sentenceRight[:4] + self.separator,
                                                         sentenceRight[4:])


def test():
    cases = [('CCSCCSCCCCCA', '|CCSCCSC|C|C|CCA|'),
             ('ECSCCCCACMCAFCCAF', '|ECSC|C|CCA|CMCAF|CCAF|'),
             ('ECSCVCC', '|ECSCV|C|C|'),
             ('ICCVCA', '|I|C|CVCA|'),
             ('CCASCCSCCVCA', '|CCASCCSC|CVCA|'),
             ('CVFCACMVVCA', '|CVFCA|CMVVCA|'),
             ('CCVGVC', '|C|CVGV|C|'),
             ('CVCCVFCV', '|CV|C|CVF|CV|'),
             ('CMMCAVCAICAF', '|CMMCAVCA|I|CAF|'),
             ('CCACMACVFCVF', '|CCACMA|CVF|CVF|'),
             ('CSCCACCACVVCA', '|CSCCA|CCA|CVVCA|'),
             ]
    tokenizer = MyanmarTokenizer()
    for case in cases:
        result = tokenizer.cutCategory(case[0])
        # Fix print statement for Python 3
        print(case, result, '|' + result + '|' == case[1])

    # Note: Removed file reading part since samples.txt may not exist
    tokenizer.separator = '@@'
    
    # Simple test case
    seg = 'စာကြည့်တိုက်'
    print(tokenizer.code2Category(seg))
    result = tokenizer.cut(seg)
    print(result)


def analyzeParams(args):
    from optparse import OptionParser
    parser = OptionParser(usage="%prog -s -i FILE or [< FILE] -o FILE or [> FILE]", version="%prog 1.0")

    parser.set_description('Syllable Segmentation Program of Myanmar\n')

    parser.add_option("-s", "--separator", dest="separator", metavar="string"
                      , help='Syllable breaking symbol, default use |', default='|')
    parser.add_option("-c", "--coding", dest="coding", metavar="string"
                      , help='Input file coding, default use utf8', default='utf8')

    parser.add_option("-i", "--input", dest="input", metavar="FILE", action="store"
                      , help='Input file(dir) path， or < FILE')
    parser.add_option("-o", "--output", dest="output", metavar="FILE", action="store"
                      , help='Output file(dir) path, or > FILE')

    (opt, args) = parser.parse_args(args)

    tokenizer = MyanmarTokenizer(opt.separator)

    if opt.input is None or os.path.isfile(opt.input):
        stdin = sys.stdin
        stdout = sys.stdout
        if opt.input is not None and os.path.exists(opt.input):
            stdin = codecs.open(opt.input, 'r', opt.coding)
        if opt.output is not None:
            stdout = codecs.open(opt.output, 'w', 'utf8')
        tokenizer.cutStd(stdin, stdout)
        if opt.input is not None and os.path.exists(opt.input):
            stdin.close()
        if opt.output is not None:
            stdout.close()
    elif os.path.isdir(opt.input):
        # Note: Removed multiprocessing part due to missing task module
        print("Directory processing requires the 'task' module which is not available")


if __name__ == "__main__":
    argv = sys.argv
    ts = time.time()
    analyzeParams(argv)
    sys.stderr.write('run time: %f\n' % (time.time()-ts))