F=open('mat_dv.txt','r')
p8=0
p9=0
p10=0
p11=0
pa=0
pg=0
for i in F:
    L=i.split()
    if L[2]=="8":
        Q=int(L[3])+int(L[4])
        if Q>p8:
            p8=Q
            N8=L
    if L[2]=="9":
        Q=int(L[3])+int(L[4])
        if Q>p9:
            p9=Q
            N9=L
    if L[2]=="10":
        Q=int(L[3])+int(L[4])
        if Q>p10:
            p10=Q
            N10=L
    if L[2]=="11":
        Q=int(L[3])+int(L[4])
        if Q>p11:
            p11=Q
            N11=L
    if int(L[3])==pa:
        pa=int(L[3])
        NA.append(L[0]+" "+L[1]+" "+ L[2])
    if int(L[3])>pa:
        pa=int(L[3])
        NA=[L[0]+" "+L[1]+" "+ L[2]]
    if int(L[4])==pg:
        pg=int(L[4])
        NG.append(L[0]+" "+L[1]+" "+ L[2])
    if int(L[4])>pg:
        pg=int(L[4])
        NG=[L[0]+" "+L[1]+" "+ L[2]]

print(N8[0], " ",N8[1]," ",p8)
print(N9[0], " ",N9[1]," ",p9)
print(N10[0], " ",N10[1]," ",p10)
print(N11[0], " ",N11[1]," ",p11)
print(NA)
print(NG)