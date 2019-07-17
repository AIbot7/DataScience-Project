from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot
import speech_recognition as sr
from espeak import espeak
import os
import time

print("Welcome .....\n1.Train data\n2.Run Program")
a = int(input("\n\nEnter your chooise\n"))
if a==1:
    while 1:
        q = input("\n\n\n\nQuestion : ")
        for f in os.listdir("files"):
            c = open("files/"+f,"r").readlines()
            if q == c :
                print("\n\nalready Exist")
            else :
                ans = input("\nanswer : ")
                with open("files/"+f,"a") as d :
                    d.write(q)
                    d.write("\n")
                    d.write(ans)
                    d.write("\n")
                    print("\n\nupdated...\n\n")
        r = input("What to continue....\n \t\t\tyes or no")
        if r == "yes" :
            continue
        else :
            break
    
elif a == 2:
    print("\n\nIn which format you like to chat \n\t\t text or speech\n ")
    w = input("\t\t\t?????????\n\t\t")
    if w == 'speech' :          
        r = sr.Recognizer()
        r.energy_threshold = 2100
        espeak.set_voice("en")
        bot = ChatBot('Test')

        bot.set_trainer(ListTrainer)

        for _files in os.listdir('files'):
            chats = open("files/"+_files,'r').readlines()

            bot.train(chats)
        print("You can ask me anything")
        while True:
                        try:
                            with sr.Microphone() as source:
                                audio = r.listen(source)
                            word = r.recognize_google(audio)
                            print("You : ",word)
                            d = bot.get_response(word)
                            s = str(d)
                            #print(type(s))
                            espeak.synth(s)
                            print("Bot : ",d)
                            #time.sleep(10)
                            continue
                        except :
                            pass
                
    elif w == 'text':
        bot = ChatBot('Test')

        bot.set_trainer(ListTrainer)

        for _files in os.listdir('files'):
            chats = open("files/"+_files,'r').readlines()

            bot.train(chats)
        print("You can ask me anything")
        while True:
            c= input("You : ")
            d = bot.get_response(c)
            s = str(d)
            #print(type(s))
            print("Bot : ",d)
    else : 
        print ("\n\n Not Match!!!!")
            
        
else :
    print ("choose one")
