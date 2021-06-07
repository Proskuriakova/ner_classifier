from corus import load_taiga_subtitles_metas, load_taiga_subtitles
from shutil import copyfile
import os


#готовит данные из корпуса Тайга - берет только русские субтитры
#параметры: 

class Prepare_Taiga_Subtitles():
    def __init__(self, text_path, subtitles_path, text_save_path, text_part: float = 0.3,
                 val_part: float = 0.2, test_part: float = 0.2):
        super(Prepare_Texts, self).__init__()
        self.text_path = text_path
        self.subtitles_path = subtitles_path
        self.text_save_path = text_save_path
        self.text_part = text_part
        self.val_part = val_part
        self.test_part = test_part
        
    def load_subtitles(self):
        path = self.text_path
        metas = load_taiga_subtitles_metas(path, offset=0, count=1)
        records = load_taiga_subtitles(path, metas, offset=0, count=1)

    def extract_russian_subtitles(self):
        # все папки с сериалами
        folder_all_subtitles = self.subtitles_path
        f_all_series = os.listdir(path=folder_all_subtitles)
        folder_all_subtitles += '/'

        folder_ru_subtitles = self.subtitles_path + '/ru_texts'
        
        f_all_ru_subtitles = []
        for f_one_series in f_all_series:
            if os.path.isdir(folder_all_subtitles+f_one_series):
        f_all_ru_subtitles_one_series = []
        f_all_subtitles_one_series = os.listdir(path=folder_all_subtitles+f_one_series)
        i = 0
        while i < len(f_all_subtitles_one_series):
            if f_all_subtitles_one_series[i].find('.ru.') != -1:
                f_all_ru_subtitles_one_series.append(f_all_subtitles_one_series[i])
                del f_all_subtitles_one_series[i]
                continue
            i += 1
        f_all_ru_subtitles.append([f_one_series, f_all_ru_subtitles_one_series])
        
        return f_all_ru_subtitles
    
    def save_russian_subtitles(self):
        
    # Копирование .txt файлов с русскими субтитрами 
        if os.path.exists(folder_ru_subtitles) == False:
            os.makedirs(folder_ru_subtitles)

        folder_ru_subtitles += '/'
        f_all_ru_subtitles = self.extract_russian_subtitles()

        for f_ru_subtitles_one_series in f_all_ru_subtitles:
            if os.path.exists(folder_ru_subtitles + f_ru_subtitles_one_series[0]) == False:
                os.makedirs(folder_ru_subtitles + f_ru_subtitles_one_series[0])
            for f_ru_subtitles_one_series_part in f_ru_subtitles_one_series[1]:
                copyfile(folder_all_subtitles + f_ru_subtitles_one_series[0] + '/' + 
                f_ru_subtitles_one_series_part, folder_ru_subtitles + f_ru_subtitles_one_series[0] + '/' + f_ru_subtitles_one_series_part)
        
    
    def get_russian_subtitles(self)
        # Получение имён всех папок с сериалами, находящимися в folder_ru_subtitles
        f_all_series = sorted(os.listdir(path = folder_ru_subtitles))[1:]

        folder_ru_subtitles += '/'

        # Получение имён всех .txt файлов с субтитрами из сериалов
        f_all_subtitles = []
        for f_one_series in f_all_series:
            f_all_subtitles.append([f_one_series, sorted(os.listdir(path=folder_ru_subtitles+f_one_series))])

        # Считывание всех .txt файлов с субтитрами из сериалов
        all_subtitles = []

        for f_subtitles_one_series in f_all_subtitles:
            for f_subtitles_one_series_part in f_subtitles_one_series[1]:
                all_subtitles_1 = []
                with open(folder_ru_subtitles+f_subtitles_one_series[0]+'/'+f_subtitles_one_series_part, 'r') as f_subtitles:
                    all_subtitles_1 += f_subtitles.readlines()
                if np.array(all_subtitles_1).size > 1:
                    all_subtitles.append(all_subtitles_1)

        return all_subtitles
    
    def clear_subtitles(self):
        
        clear_subtitles = []
        clear_sentence = []
        all_subtitles = self.get_russian_subtitles()
        
        for i in range(len(all_subtitles)):
            for j in range(len(all_subtitles[i])):
                sentence = all_subtitles[i][j]
                
                sentence = re.sub(r'(\ufeff)?\d+\t?\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}\t?\d{1,3}:\d{1,2}:\d{1,2},\d{1,5}\t?', '', sentence) 
                # Удаление временной метки внутри строки с разбиением строки на отдельные предложения
                coincidence = re.findall(r'\d{1,3}\s*\t?\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}\s*-?-?>?\s*\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}', sentence) 
                if len(coincidence) != 0:
                    index = sentence.find(coincidence[0])
                    all_subtitles[i].insert(j + 1, sentence[index + len(coincidence[0]):])
                    sentence = sentence[:index]

                sentence += ' ' 
                sentence = ' ' + sentence # Что бы дефис в самом начале корректно обрабатывался (если он есть)

                # Удаление ссылок
                sentence = re.sub(r'(www\.[^\s]+)|(https?://[^\s]+)|(\.com)|(\.ru)|(\.org)', ' ', sentence)

                # Удаление скобок вместе с их содержимым
                sentence = re.sub(r'\([^)]*\)', '', sentence)
                sentence = re.sub(r'\[.*?\]', '', sentence)

                # Добавление пробелов перед тире и замена его на дефис
                sentence = re.sub(r'-+', '-', sentence)
                sentence = re.sub(r'[\d\s\.,]-', ' - ', sentence)
                sentence = re.sub(r'[\s\.,]—', ' - ', sentence)
                sentence = re.sub(r'—', '-', sentence)

                # Удаление html-тегов
                sentence = re.sub(r'<[^>]*>', '', sentence)

                # Удаление всего, что не является русской/английской буквой, цифрой, '!', '?', ',', '.', ':', '*' или '-'
                sentence = re.sub(r'[^a-zA-Zа-яА-Я0-9!\?,\.:\*]+', ' ', sentence)

                # Удаление номера сезона и эпизода
                sentence = re.sub(r'сезон\s\d+\sэпизод\s\d+', '', sentence)

                # Замена нескольких подряд идущих ','  '!' и '?' на одиночные
                sentence = re.sub(r',+', ',', sentence)
                sentence = re.sub(r'!+', '!', sentence)
                sentence = re.sub(r'\?+', '?', sentence)

                # Удаление пробелов перед и/или после '…'
                sentence = re.sub(r'\s?(\.{2,10}\s?)+', '… ', sentence)
                sentence = re.sub(r'\s…', '…', sentence)

                # Замена конструкций '?...' и '!...' на '?' и '!' 
                sentence = re.sub(r'\?…', '?', sentence)
                sentence = re.sub(r'!…', '!', sentence)

                # Удаление пробелов перед и/или после ',', '.', '!' и '?'
                sentence = re.sub(r'\s?,\s?', ', ', sentence)
                sentence = re.sub(r'\s?\.\s?', '. ', sentence)
                sentence = re.sub(r'\s?!\s?', '! ', sentence)
                sentence = re.sub(r'\s?\?\s?', '? ', sentence)

                # Исправление конструкций '!,', '!.', '?,', '?.', ',.', '.,', ',!'. '.!', ',?', '.?'
                sentence = re.sub(r'!\s*,', '!', sentence)
                sentence = re.sub(r'!\s*\.', '!', sentence)
                sentence = re.sub(r'\?\s*,', '?', sentence)
                sentence = re.sub(r'\?\s*[\.…]', '?', sentence)
                sentence = re.sub(r',\s*[\.…]', ',', sentence)
                sentence = re.sub(r'\.\s*,', '.', sentence)
                sentence = re.sub(r'…\s*,', '…', sentence)
                sentence = re.sub(r',\s*!', ',', sentence)
                sentence = re.sub(r'\.\s*!', '.', sentence)
                sentence = re.sub(r',\s*\?', ',', sentence)
                sentence = re.sub(r'\.\s*\?', '.', sentence)

                # Удаление пробела после '.' или ',' в дробных числах
                sentence = re.sub(r'(\d+)[,.]\s(\d+)', r'\1,\2', sentence)

                # Удаление нескольких подряд идущих пробелов
                sentence = re.sub(r'\s+', ' ', sentence)

                # Удаление пробелов в начале и конце строки
               
                sentence = sentence.strip() 

                all_subtitles[i][j] = sentence
                
        for i in range(len(all_subtitles)):
            sentences = all_subtitles[i]
            lexicon = ['', 'Переведено', 'Переводчики:']
            lexicon = set(lexicon)
            all_subtitles[i] = [s for s in sentences if not any(w in lexicon for w in s.split())]
            
        return all_subtitles
                
        
    def markup_subtitles(self):
        subtitles = self.clear_subtitles()
        markup_text = []
        chars = [',', '.', '?', '!']
        for text in subtitles:
            for sent in text:
                for word in sent.split():
                    label = ''
                    if word[0].isupper():
                        label = 'caps-'
                    else:
                        label = 'o-'
                    if word[-1] == '.':
                        label = label + 'dot'
                    if word[-1] == ',':
                        label = label + 'comma'
                    if word[-1] == '?':
                        label = label + 'question'
                    if word[-1] == '!':
                        label = label + 'exclamation'
                    if word[-1] not in chars:
                        label = label + 'o'
                    markup_text.append(word + ' ' + label)

                markup_text.append('\n')

        return markup_text
    
    def save_marked_up(self):
        
        text = self.markup_subtitles()
        text_to_save = text[:int(len(text) * self.text_part)]
        train_part = 1.0 - self.val_part - self.test_path
        train = text_to_save[:int(len(text_to_save) * train_part)]
        val = text_to_save[int(len(text_to_save) * train_part) + 1 : int(len(text_to_save) * (train_part + self.val_part))]
        test = text_to_save[int(len(text_to_save) * (train_part + self.val_part)) + 1 :]
        
        #для теста убираем разметку, разметку записываем отдельно
        test_texts, test_targets = [], []
        for item in test[0]:
            if item != '\n':
                test_texts.append(item.split('\t')[0])
                test_targets.append(item.split('\t')[1].replace('\n', ''))
        
        
        with open(self.text_save_path + '/train.txt', 'w') as f:
            for item in train:
                f.write("%s\n" % item)
        
        with open(self.text_save_path + '/val.txt', 'w') as f:
            for item in val:
                f.write("%s\n" % item)
                
        with open(self.text_save_path + '/test.txt', 'w') as f:
            for item in test_texts:
                f.write("%s\n" % item)
                
        with open(self.text_save_path + '/test_target.txt', 'w') as f:
            for item in test_targets:
                f.write("%s\n" % item)
        
    

#для любого текста: токенизирует, исправляет тексты (удаляет временные метки, html-теги), формирует разметку для обучения и сохраняет датасет: отдельные файлы train.txt, val.txt, test.txt и test_target.txt
#в качестве параметров принимает:
#text_path - путь до текста
#text_save_path - путь до папки, куда будет сохранен датасет
#text_part - какую часть всего текста использовать, по умолчанию = 1.0
#val_part - доля датасета для валидации, по умолчанию = 0.2
#test_part - доля датасета для теста, по умолчанию = 0.1

class Prepare_Text():
    def __init__(self, text_path, text_save_path, text_part: float = 1.0,
                 val_part: float = 0.2, test_part: float = 0.1):
        super(Prepare_Texts, self).__init__()
        self.text_path = text_path
        self.text_save_path = text_save_path
        self.text_part = text_part
        self.val_part = val_part
        self.test_part = test_part
        

    def clear_text(self):
        
        clear_subtitles = []
        clear_sentence = []
        all_subtitles = self.get_russian_subtitles()
        
        for i in range(len(all_subtitles)):
            for j in range(len(all_subtitles[i])):
                sentence = all_subtitles[i][j]
                
                sentence = re.sub(r'(\ufeff)?\d+\t?\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}\t?\d{1,3}:\d{1,2}:\d{1,2},\d{1,5}\t?', '', sentence) 
                # Удаление временной метки внутри строки с разбиением строки на отдельные предложения
                coincidence = re.findall(r'\d{1,3}\s*\t?\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}\s*-?-?>?\s*\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}', sentence) 
                if len(coincidence) != 0:
                    index = sentence.find(coincidence[0])
                    all_subtitles[i].insert(j + 1, sentence[index + len(coincidence[0]):])
                    sentence = sentence[:index]

                sentence += ' ' 
                sentence = ' ' + sentence # Что бы дефис в самом начале корректно обрабатывался (если он есть)

                # Удаление ссылок
                sentence = re.sub(r'(www\.[^\s]+)|(https?://[^\s]+)|(\.com)|(\.ru)|(\.org)', ' ', sentence)

                # Удаление скобок вместе с их содержимым
                sentence = re.sub(r'\([^)]*\)', '', sentence)
                sentence = re.sub(r'\[.*?\]', '', sentence)

                # Добавление пробелов перед тире и замена его на дефис
                sentence = re.sub(r'-+', '-', sentence)
                sentence = re.sub(r'[\d\s\.,]-', ' - ', sentence)
                sentence = re.sub(r'[\s\.,]—', ' - ', sentence)
                sentence = re.sub(r'—', '-', sentence)

                # Удаление html-тегов
                sentence = re.sub(r'<[^>]*>', '', sentence)

                # Удаление всего, что не является русской/английской буквой, цифрой, '!', '?', ',', '.', ':', '*' или '-'
                sentence = re.sub(r'[^a-zA-Zа-яА-Я0-9!\?,\.:\*]+', ' ', sentence)

                # Замена нескольких подряд идущих ','  '!' и '?' на одиночные
                sentence = re.sub(r',+', ',', sentence)
                sentence = re.sub(r'!+', '!', sentence)
                sentence = re.sub(r'\?+', '?', sentence)

                # Удаление пробелов перед и/или после '…'
                sentence = re.sub(r'\s?(\.{2,10}\s?)+', '… ', sentence)
                sentence = re.sub(r'\s…', '…', sentence)

                # Замена конструкций '?...' и '!...' на '?' и '!' 
                sentence = re.sub(r'\?…', '?', sentence)
                sentence = re.sub(r'!…', '!', sentence)

                # Удаление пробелов перед и/или после ',', '.', '!' и '?'
                sentence = re.sub(r'\s?,\s?', ', ', sentence)
                sentence = re.sub(r'\s?\.\s?', '. ', sentence)
                sentence = re.sub(r'\s?!\s?', '! ', sentence)
                sentence = re.sub(r'\s?\?\s?', '? ', sentence)

                # Исправление конструкций '!,', '!.', '?,', '?.', ',.', '.,', ',!'. '.!', ',?', '.?'
                sentence = re.sub(r'!\s*,', '!', sentence)
                sentence = re.sub(r'!\s*\.', '!', sentence)
                sentence = re.sub(r'\?\s*,', '?', sentence)
                sentence = re.sub(r'\?\s*[\.…]', '?', sentence)
                sentence = re.sub(r',\s*[\.…]', ',', sentence)
                sentence = re.sub(r'\.\s*,', '.', sentence)
                sentence = re.sub(r'…\s*,', '…', sentence)
                sentence = re.sub(r',\s*!', ',', sentence)
                sentence = re.sub(r'\.\s*!', '.', sentence)
                sentence = re.sub(r',\s*\?', ',', sentence)
                sentence = re.sub(r'\.\s*\?', '.', sentence)

                # Удаление пробела после '.' или ',' в дробных числах
                sentence = re.sub(r'(\d+)[,.]\s(\d+)', r'\1,\2', sentence)

                # Удаление нескольких подряд идущих пробелов
                sentence = re.sub(r'\s+', ' ', sentence)

                # Удаление пробелов в начале и конце строки
               
                sentence = sentence.strip() 

                all_subtitles[i][j] = sentence
                
            
        return all_subtitles
                
        
    def markup_subtitles(self):
        subtitles = self.clear_subtitles()
        markup_text = []
        chars = [',', '.', '?', '!']
        for text in subtitles:
            for sent in text:
                for word in sent.split():
                    label = ''
                    if word[0].isupper():
                        label = 'caps-'
                    else:
                        label = 'o-'
                    if word[-1] == '.':
                        label = label + 'dot'
                    if word[-1] == ',':
                        label = label + 'comma'
                    if word[-1] == '?':
                        label = label + 'question'
                    if word[-1] == '!':
                        label = label + 'exclamation'
                    if word[-1] not in chars:
                        label = label + 'o'
                    markup_text.append(word + ' ' + label)

                markup_text.append('\n')

        return markup_text
    
    def save_marked_up(self):
        
        text = self.markup_subtitles()
        text_to_save = text[:int(len(text) * self.text_part)]
        train_part = 1.0 - self.val_part - self.test_path
        train = text_to_save[:int(len(text_to_save) * train_part)]
        val = text_to_save[int(len(text_to_save) * train_part) + 1 : int(len(text_to_save) * (train_part + self.val_part))]
        test = text_to_save[int(len(text_to_save) * (train_part + self.val_part)) + 1 :]
        
        #для теста убираем разметку, разметку записываем отдельно
        test_texts, test_targets = [], []
        for item in test[0]:
            if item != '\n':
                test_texts.append(item.split('\t')[0])
                test_targets.append(item.split('\t')[1].replace('\n', ''))
        
        
        with open(self.text_save_path + '/train.txt', 'w') as f:
            for item in train:
                f.write("%s\n" % item)
        
        with open(self.text_save_path + '/val.txt', 'w') as f:
            for item in val:
                f.write("%s\n" % item)
                
        with open(self.text_save_path + '/test.txt', 'w') as f:
            for item in test_texts:
                f.write("%s\n" % item)
                
        with open(self.text_save_path + '/test_target.txt', 'w') as f:
            for item in test_targets:
                f.write("%s\n" % item)
        

