golden_file = "./dev_other_qtype.txt"
pre_file = "./dev_ent_qtype50.txt"

f =open("./dev_other-ans-style.txt", 'a', encoding="utf-8")
i=0
k=0
with open(golden_file, 'r', encoding="utf-8") as gf, open(pre_file, 'r', encoding='utf-8') as pf:
    for g, p in zip(gf, pf):
        g = g.strip()
        p = p.strip().split()
        if g[0] == "other":
            f.write(g[0] + "\r\n")
        else:
            f.write(p + "\r\n")


f.close()