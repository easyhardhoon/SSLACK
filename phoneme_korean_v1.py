from collections import deque
from logging import exception
#import speech_recognition as sr 
#from py_hanspell.hanspell import spell_checker  # 맞춤법 검사기
from gtts import gTTS
import os
import time
import playsound
from py_hanspell.hanspell import spell_checker

class Phoneme_korean:
    '''한글 전용 수어 클래스'''
    first = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
             "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]          # 초성
    second = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
              "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]   # 중성
    third = [" ", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", 
             "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", 
             "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]               # 종성
    exception = {"ㄱ+ㅅ":"ㄳ", "ㄴ+ㅈ":"ㄵ", "ㄴ+ㅎ":"ㄶ", "ㄹ+ㄱ":"ㄺ",
                 "ㄹ+ㅁ":"ㄻ", "ㄹ+ㅂ":"ㄼ", "ㄹ+ㅅ":"ㄽ", "ㄹ+ㅌ":"ㄾ",
                 "ㄹ+ㅍ":"ㄿ", "ㄹ+ㅎ":"ㅀ", "ㅂ+ㅅ":"ㅄ"}                  # 종성 이중 받침
    after_spell_check = ""
    
    def __init__(self, input_phoneme:list):
        '''음소 셋 리스트 생성 메소드'''
        self.phoneme = [Phoneme_korean.first, Phoneme_korean.second, Phoneme_korean.third]
        self.phoneme_queue = deque(input_phoneme)
        
        self.__sentence_list = []
        self.__unicode_sum = 44032
        self.__phoneme_count = 0
    
    def merge_phoneme(self):
        '''음소 조합 메소드'''
        #['ㄱ', 'ㅏ', 'ㅇ', 'ㅅ', 'ㅡ', 'ㅇ', 'ㅁ', 'ㅣ', 'ㄴ']
        self.phoneme_queue = self.convert_dot(self.phoneme_queue)
        while(1):
            target = self.phoneme_queue.popleft()
            for i in range(len(self.phoneme[self.__phoneme_count])):
                if self.phoneme[self.__phoneme_count][i] == target:
                    if self.__phoneme_count == 0:
                        self.__unicode_sum += 588 * i
                        self.__phoneme_count += 1
                        break
                    
                    elif self.__phoneme_count == 1:
                        self.__unicode_sum += 28 * i
                        self.__phoneme_count += 1

                        if len(self.phoneme_queue) > 1:
                            if self.phoneme_queue[1] in self.phoneme[1]:
                                self.__initial_phoneme_count()
                                self.__sentence_list.append(chr(self.__unicode_sum))
                                self.__initial_unicode_sum()
                        break
                    
                    elif self.__phoneme_count == 2:
                        if len(self.phoneme_queue) == 0:
                            #마지막에 종성 한개만 올 경우
                            self.__unicode_sum += i
                            self.__sentence_list.append(chr(self.__unicode_sum))
                            self.__initial_unicode_sum()
                            self.__initial_phoneme_count()
                            break
                        
                        elif len(self.phoneme_queue) == 1:
                            #마지막에 종성 두개가 올 경우 ex) ㄱ + ㅅ
                            temp = target + "+" + self.phoneme_queue.popleft()
                            self.__unicode_sum += self.phoneme[2].index(Phoneme_korean.exception[temp])
                            self.__sentence_list.append(chr(self.__unicode_sum))
                            self.__initial_unicode_sum()
                            self.__initial_phoneme_count()
                            break
                            
                        else:
                            if self.phoneme_queue[1] in self.phoneme[1]:
                                self.__unicode_sum += i
                                self.__sentence_list.append(chr(self.__unicode_sum))
                                self.__initial_unicode_sum()
                                self.__initial_phoneme_count()
                                break
                            elif self.phoneme_queue[1] in self.phoneme[0]:
                                temp = target + "+" + self.phoneme_queue.popleft()
                                self.__unicode_sum += self.phoneme[2].index(Phoneme_korean.exception[temp])
                                self.__sentence_list.append(chr(self.__unicode_sum))
                                self.__initial_unicode_sum()
                                self.__initial_phoneme_count()
                                break

            if len(self.phoneme_queue) == 0:
                if self.__phoneme_count != 0:
                    self.__sentence_list.append(chr(self.__unicode_sum))
                break

    @property
    def sentence_list(self):
        """음소 문장 getter 메소드"""
        return self.__sentence_list

    def print_sentence(self):
        '''리스트 출력 메소드'''
        ret = ""
        for i in self.__sentence_list:
            ret += i
        print("완성된 문장: " + ret)

    def convert_dot(self, phoneme_queue):
        '''. + 자음을 된소리로 변환해주는 메소드'''
        phoneme_list = list(phoneme_queue)
        phoneme_dict = {"ㄱ" : 12594, "ㄷ" : 12600, "ㅂ" : 12611, "ㅅ" : 12614, "ㅈ" : 12617}
        i = 0

        while(1):
            if i == len(phoneme_list):
                break
        
            if phoneme_list[i] == ".":
                phoneme_list[i - 1] = chr(phoneme_dict[phoneme_list[i - 1]])
                phoneme_list.pop(i)
        
            else:
                i += 1
        
        return deque(phoneme_list)
        
    def __initial_unicode_sum(self):
        '''유니코드 합 초기화 메소드'''
        self.__unicode_sum = 44032
    
    def __initial_phoneme_count(self):
        '''음소 카운트 초기화 메소드'''
        self.__phoneme_count = 0

    def speak(self, text):
        #TTS 메소드
        #text = "".join(list)
        tts = gTTS(text=text, lang="ko")
        filename='voice2.mp3'
        tts.save(filename)
        playsound.playsound(filename)

    def test(self):
        self.merge_phoneme()
        self.print_sentence()

        print("맞춤법 검사 전:", self)

        # 맞춤법 검사기
        sentence = str(self)
        spelled_sentence = spell_checker.check(sentence)
       
        print("맞춤법 검사 후:", spelled_sentence)
        Phoneme_korean.after_spell_check = spelled_sentence
        # TTS
        self.speak(spelled_sentence)
         

    def __str__(self):
        '''문자열 출력 메소드'''
        ret = ""
        for i in self.__sentence_list:
            ret += i
        return ret

    def get_str(self):
        """맞춤법 검사 후 문자열 출력 메소드"""
        return Phoneme_korean.after_spell_check
    
    def get_list(self):
        return self.__sentence_list