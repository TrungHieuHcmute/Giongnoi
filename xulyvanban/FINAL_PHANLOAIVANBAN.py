import regex as re
from underthesea import word_tokenize
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
dicchar = loaddicchar()
def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)
bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']
nguyen_am_to_ids = {}
for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)
def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word
    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:
            chars[index] = bang_nguyen_am[x][dau_cau]
            for index2 in nguyen_am_index:
                if index2 != index:
                    x, y = nguyen_am_to_ids[chars[index]]
                    chars[index2] = bang_nguyen_am[x][0]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return ''.join(chars)
def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True

def chuan_hoa_dau_cau_tieng_viet(sentence):
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

def remove_html(txt):
    return re.sub(r'<[^>]*>', '', txt)
def text_preprocess(document):
    # xóa html code
    document = remove_html(document)
    # chuẩn hóa unicode
    document = convert_unicode(document)
    # chuẩn hóa cách gõ dấu tiếng Việt
    document = chuan_hoa_dau_cau_tieng_viet(document)
    # tách từ
    document = word_tokenize(document, format="text")
    # đưa về lower
    document = document.lower()
    # xóa các ký tự không cần thiết
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()
    return document
######Thống kê số lượng data theo nhãn##############################################################################################################################################

count = {}
for line in open('news_categories.txt', encoding='utf-8'):
    key = line.split()[0]
    count[key] = count.get(key, 0) + 1
for key in count:
    print(key, count[key])

######Thống kê các word xuất hiện ở tất cả các nhãn##############################################################################################################################################

# total_label = 8
# vocab = {}
# label_vocab = {}
# for line in open('news_categories.txt', encoding='utf-8'):
#     words = line.split()
#     # lưu ý từ đầu tiên là nhãn
#     label = words[0]
#     if label not in label_vocab:
#         label_vocab[label] = {}
#     for word in words[1:]:
#         label_vocab[label][word] = label_vocab[label].get(word, 0) + 1
#         if word not in vocab:
#             vocab[word] = set()
#         vocab[word].add(label)
# count = {}
# for word in vocab:
#     for label in label_vocab:
#         if word in label_vocab[label]:
#             count[word] = count.get(word, 0) + label_vocab[label][word]
# sorted_count = sorted(count, key=count.get, reverse=True)
# for word in sorted_count[:1000]:
#     print(word, count[word])

######loại stopword khỏi dữ liệu (loại những từ xuất hiện nhiều) ##############################################################################################################################################

# stopword = set()
# with open('stopwords.txt', 'w', encoding='utf-8') as fp:
#     for word in sorted_count[:1000]:
#         stopword.add(word)
#         fp.write(word + '\n')
# def remove_stopwords(line):
#     words = []
#     for word in line.strip().split():
#         if word not in stopword:
#             words.append(word)
#     return ' '.join(words)
# with open('news_categories.prep', 'w', encoding='utf-8') as fp:
#     for line in open('news_categories.txt', encoding='utf-8'):
#         line = remove_stopwords(line)
#         fp.write(line + '\n')

######### Chia tập train/test####################################################################################################################################################################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
test_percent = 0.2
text = []
label = []
for line in open('news_categories.txt', encoding='utf-8'):
    words = line.strip().split()
    label.append(words[0])
    text.append(' '.join(words[1:]))
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=50)
# Lưu train/test data
# Giữ nguyên train/test để về sau so sánh các mô hình cho công bằng
with open('train.txt', 'w', encoding='utf-8') as fp:
    for x, y in zip(X_train, y_train):
        fp.write('{} {}\n'.format(y, x))

with open('test.txt', 'w', encoding='utf-8') as fp:
    for x, y in zip(X_test, y_test):
        fp.write('{} {}\n'.format(y, x))
################encode label#######################################################################################################################################
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
# In ra ánh xạ giữa nhãn và số nguyên tương ứng
label_mapping = {label: encoded for label, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
# print(label_mapping)
# print(list(label_encoder.classes_), '\n')
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
# print(X_train[0], y_train[0], '\n')
# print(X_test[0], y_test[0])
###############################################################################
MODEL_PATH = "models"
import os
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
import pickle
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
start_time = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1),
                                             max_df=0.8,
                                             max_features=None)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())
                    ])
text_clf = text_clf.fit(X_train, y_train)
train_time = time.time() - start_time
print('Done training Naive Bayes in', train_time, 'seconds.')
# Save model
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'wb'))
####### Naive Bayes#########################################################################
import numpy as np
model = pickle.load(open(os.path.join(MODEL_PATH,"naive_bayes.pkl"), 'rb'))
y_pred = model.predict(X_test)
print('Naive Bayes, Accuracy =', np.mean(y_pred == y_test))
# Xem kết quả trên từng nhãn
nb_model = pickle.load(open(os.path.join(MODEL_PATH,"naive_bayes.pkl"), 'rb'))
#######################################################################################################################################################
####################################GIONG NOI###############################################################################
#######################################################################################################################################################
import speech_recognition as sr
from gtts import gTTS
import os
import time
import playsound
import pyaudio

while True:
    document = input(" nhap  ")
    # document = text_preprocess(document)
    label = nb_model.predict([document])
    print('Predict label:', label_encoder.inverse_transform(label))
    #     r = sr.Recognizer()
    #     def speak(text):
    #         tts = gTTS(text=text, lang='vi')
    #         filename = "voice.mp3"
    #         tts.save(filename)
    #         playsound.playsound(filename)
    #         os.remove(filename)
    #
    #
    #     with sr.Microphone() as source:
    #         r.adjust_for_ambient_noise(source)
    #         audio_data = r.record(source, duration=5)
    #         print("Recognizing ...")
    #     try:
    #         text = r.recognize_google(audio_data, language="vi")
    #     except:
    #         text = ""
    #     if (text) != "":
    #         print(text)
    #         document = text
    #         document = text_preprocess(document)
    #         label = nb_model.predict([document])
    #         print('Predict label:', label_encoder.inverse_transform(label))
    #         if (label_encoder.inverse_transform(label)) == "__label__bắt_đầu" or "bắt đầu" in text:
    #             speak("Bắt đầu")
    #         if (label_encoder.inverse_transform(label)) == "__label__trở_về" or "trở về" in text:
    #             speak("Trở về")
    #         if (label_encoder.inverse_transform(label)) == "__label__tra_cứu" or "tra cứu" in text:
    #             speak("Tra cứu")
    #         if (label_encoder.inverse_transform(label)) == "__label__mượn_sách" or "mượn" in text:
    #             speak("Mượn sách")
    #         if (label_encoder.inverse_transform(label)) == "__label__đồng_ý":
    #             speak("Đồng ý")
    #         if (label_encoder.inverse_transform(label)) == "__label__không_đồng_ý" or " không đồng ý" in text:
    #             speak("Không đồng ý")
    #         if (label_encoder.inverse_transform(label)) == "__label__thoát_ra" or "thoát" in text:
    #             speak("Thoát ra")
    #         if (label_encoder.inverse_transform(label)) == "__label__trả_sách" or "trả sách" in text:
    #             speak("Trả sách")
    #         if (label_encoder.inverse_transform(label)) == "__label__vào_học" or "vào học" in text:
    #             speak("Vào học")






