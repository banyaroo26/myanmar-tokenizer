# myanmar-tokenizer
A Rule-based Syllable Segmentation of Myanmar Text

```
from myanmar_tokenizer import p3_tokenizer

tk = p3_tokenizer.MyanmarTokenizer(separator=" ")

sentence = "ကျောင်းအုပ်ဆရာကြီး"

sentence = tk.cut(sentence).replace('?', ' ')

print(sentence) # ကျောင်း အုပ် ဆ ရာ ကြီး
```
