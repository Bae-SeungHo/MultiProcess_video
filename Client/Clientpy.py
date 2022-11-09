from multiprocessing import Process , Queue , Manager
import Source as S

def main():
        
    manager = Manager()
    OnlineQ = manager.Queue()
    #GestureQ = manager.Queue()
    #SoundQ = manager.Queue()
    server = S.Connect(9925)
    T_Thread = Process(target=S.T_Thread, args=(server,OnlineQ))
    T_Thread.start()

#R_Thread = Process(target=S.R_Thread, args=(server,SoundQ,OnlineQ))
#R_Thread.start()  

#Sound_Thread = Process(target=S.Sound_Thread, args=(SoundQ,OnlineQ))  
#Sound_Thread.start()  

if __name__ == '__main__':
    main()
