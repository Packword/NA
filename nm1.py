file=open(r"D:\Zhelnin\travels.txt","r")
p1=0
p2=0
p3=0
p4=0
p5=0
j=0
s=set()
s1=set()
d={}
d1={}
for line in file:
    qu=line.split()
    s.add(qu[2])
    s1.add(qu[3])
    if qu[0]=="1":
        p1+=int(qu[6])
        p5+=int(qu[4])
    if qu[0]=="2":
        p2+=int(qu[6])
    if qu[0]=="3":
        p3+=int(qu[6])
    if qu[2]=="Липки" :
        p4+=int(qu[6])
file.close()
file=open(r"D:\Zhelnin\travels.txt","r")
for i in s:
    d[i]=0
    d1[i]=0
for line in file:
    qu=line.split()
    pr=d.get(qu[3])
    d[qu[3]]=pr+int(qu[5])
    d1[qu[3]]+=1
for i in d:
    w=d.get(i)
    w1=d1.get(i)
    w2=w/w1
    if w2>j:
        j=w2
L=[p1,p2,p3]
d=max(L)
print(d)
print(L.index(d))
print(p4)
print(p5)
print(s)
print(s1)
print(j)
file.close()