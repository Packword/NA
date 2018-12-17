file=open(r"D:\Zhelnin\perepis.txt","r")
l=0
list=[]
a,b=map(int,input().split())
for line in file:
    qu=line.split(".")
    o=qu[0].split()
    z=int(qu[2][0:-1])
    if z<1978:
        l+=1
        print(o[0])
    if z>=a and z<=b:
        list.append(line[:-1])
print(l)
print(list)


file.close()