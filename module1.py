def Uscor (V,VE,T):
    print((VE-V)/T)
def Decorate(Uscor,V,VE,T):
    def wrapper():
        Uscor(V,VE,T)
        print('S= ', ((V+VE)/2)*T)
    return wrapper
Uscor=Decorate(Uscor,2,4,2)
Uscor()