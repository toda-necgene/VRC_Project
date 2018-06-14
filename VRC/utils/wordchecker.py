import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
word={ "あ": "a" , "い": "i","う": "u","え" : "e" ,"お" : "o" ,
       "か":"ka" , "き":"ki","く":"ku","け" :"ke" ,"こ" :"ko" ,
       "さ":"sa" , "し":"si","す":"su","せ" :"se" ,"そ" :"so" ,
       "た":"ta" , "ち":"ti","つ":"tu","て" :"te" ,"と" :"to" ,
       "な":"na" , "に":"ni","ぬ":"nu","ね" :"ne" ,"の" :"no" ,
       "は":"ha" , "ひ":"hi","ふ":"hu","へ" :"he" ,"ほ" :"ho" ,
       "ま":"ma" , "み":"mi","む":"mu","め" :"me" ,"も" :"mo" ,
       "ら":"ra" , "り":"ri","る":"ru","れ" :"re" ,"ろ" :"ro" ,
       "が":"ga" , "ぎ":"gi","ぐ":"gu","げ" :"ge" ,"ご" :"go" ,
       "ざ":"za" , "じ":"zi","ず":"zu","ぜ" :"ze" ,"ぞ" :"zo" ,
       "だ":"da" , "ぢ":"zi","づ":"du","で" :"de" ,"ど" :"do" ,
       "ば":"ba" , "び":"bi","ぶ":"bu","べ" :"be" ,"ぼ" :"bo" ,
       "ぱ":"pa" , "ぴ":"pi","ぷ":"pu","ぺ" :"pe" ,"ぽ" :"po" ,
       "や":"ja" , "ぃ":"ii","ゆ":"ju","いぇ" :"je" ,"よ" :"jo" ,
       "わ":"wa" , "うぃ":"wi","ぅ":"wu","うぇ" :"we" ,"を" :"wo" ,
       "きゃ":"kRa" , "きぃ":"kRi","きゅ":"kRu","きぇ" :"kRe" ,"きょ" :"kRo" ,
       "しゃ":"sRa" , "しぃ":"sRi","しゅ":"sRu","しぇ" :"sRe" ,"しょ" :"sRo" ,
       "ちゃ":"tRa" , "ちぃ":"tRi","ちゅ":"tRu","ちぇ" :"tRe" ,"ちょ" :"tRo" ,
       "にゃ":"nRa" , "にぃ":"nRi","にゅ":"nRu","にぇ" :"nRe" ,"にょ" :"nRo" ,
       "ひゃ":"hRa" , "ひぃ":"hRi","ひゅ":"hRu","ひぇ" :"hRe" ,"ひょ" :"hRo" ,
       "ふぁ":"fa" , "ふぃ":"fi","ふゅ":"hu","ふぇ" :"fe" ,"ふぉ" :"fo" ,
       "みゃ":"mRa" , "みぃ":"mRi","みゅ":"mRu","みぇ" :"mRe" ,"みょ" :"mRo" ,
       "りゃ":"rRa" , "りぃ":"rRi","りゅ":"rRu","りぇ" :"rRe" ,"りょ" :"rRo" ,
       "ぎゃ":"gRa" , "ぎぃ":"gRi","ぎゅ":"gRu","ぎぇ" :"gRe" ,"ぎょ" :"gRo" ,
       "じゃ":"zRa" , "じぃ":"zRi","じゅ":"zRu","じぇ" :"zRe" ,"じょ" :"zRo" ,
       "ぢゃ":"dRa" , "ぢぃ":"dRi","ぢゅ":"dRu","ぢぇ" :"dRe" ,"ぢょ" :"dRo" ,
       "びゃ":"bRa" , "びぃ":"bRi","びゅ":"bRu","びぇ" :"bRe" ,"びょ" :"bRo" ,
       "ぴゃ":"pRa" , "ぴぃ":"pRi","ぴゅ":"pRu","ぴぇ" :"pRe" ,"ぴょ" :"pRo" ,
       "ア": "a" , "イ": "i","ウ": "u","エ" : "e" ,"オ" : "o" ,
       "カ":"ka" , "キ":"ki","ク":"ku","ケ" :"ke" ,"コ" :"ko" ,
       "サ":"sa" , "シ":"si","ス":"su","セ" :"se" ,"ソ" :"so" ,
       "タ":"ta" , "チ":"ti","ツ":"tu","テ" :"te" ,"ト" :"to" ,
       "ナ":"na" , "ニ":"ni","ヌ":"nu","ネ" :"ne" ,"ノ" :"no" ,
       "ハ":"ha" , "ヒ":"hi","フ":"hu","ヘ" :"he" ,"ホ" :"ho" ,
       "マ":"ma" , "ミ":"mi","ム":"mu","メ" :"me" ,"モ" :"mo" ,
       "ラ":"ra" , "リ":"ri","ル":"ru","レ" :"re" ,"ロ" :"ro" ,
       "ガ":"ga" , "ギ":"gi","グ":"gu","ゲ" :"ge" ,"ゴ" :"go" ,
       "ザ":"za" , "ジ":"zi","ズ":"zu","ゼ" :"ze" ,"ゾ" :"zo" ,
       "ダ":"da" , "ヂ":"zi","ヅ":"du","デ" :"de" ,"ド" :"do" ,
       "バ":"ba" , "ビ":"bi","ブ":"bu","ベ" :"be" ,"ボ" :"bo" ,
       "パ":"pa" , "ピ":"pi","プ":"pu","ペ" :"pe" ,"ポ" :"po" ,
       "ヤ":"ja" , "ィ":"ji","ユ":"ju","イェ" :"je" ,"ヨ" :"jo" ,
       "ワ":"wa" , "ウィ":"wi","ッ":"wu","ウェ" :"we" ,"ヲ" :"wo" ,
       "キャ":"kRa" , "キィ":"kRi","キュ":"kRu","キェ" :"kRe" ,"キョ" :"kRo" ,
       "シャ":"sRa" , "シィ":"sRi","シュ":"sRu","シェ" :"sRe" ,"ショ" :"sRo" ,
       "チャ":"tRa" , "チィ":"tRi","チュ":"tRu","チェ" :"tRe" ,"チョ" :"tRo" ,
       "ニャ":"nRa" , "ニィ":"nRi","ニュ":"nRu","ニェ" :"nRe" ,"ニョ" :"nRo" ,
       "ヒャ":"hRa" , "ヒィ":"hRRi","ヒュ":"hRu","ヒェ" :"hRe" ,"ヒョ" :"hRo" ,
       "ファ":"fa" , "フィ":"fi","フュ":"fu","ふぇ" :"fe" ,"フォ" :"fo" ,
       "ミャ":"mRa" , "ミィ":"mRi","ミュ":"mRu","ミェ" :"mRe" ,"ミョ" :"mRo" ,
       "リャ":"rRa" , "リィ":"rRi","ヂュ":"rRu","リェ" :"rRe" ,"リョ" :"rRo" ,
       "ギャ":"gRa" , "ギィ":"gRi","ギュ":"gRu","ギェ" :"gRe" ,"ギョ" :"gRo" ,
       "ジャ":"zRa" , "ジィ":"zRi","ジュ":"zRu","ジェ" :"zRe" ,"ジョ" :"zRo" ,
       "ヂャ":"dRa" , "ヂィ":"dRi","ヂュ":"dRu","ヂェ" :"dRe" ,"ヂョ" :"dRo" ,
       "ビャ":"bRa" , "ビィ":"bRi","ビュ":"bRu","ビェ" :"bRe" ,"ビョ" :"bRo" ,
       "ピャ":"pRa" , "ピィ":"pRi","ピュ":"pRu","ピェ" :"pRe" ,"ピョ" :"pRo" ,
       "ッ":"H","ー":"Q","ン":"N","っ":"Q","ん":"N","\n":"?","、":"?","。":"?"," ":"?","　":"?","":"?"}
bo=["a","i","u","e","o"]
si=["k","s","t","n","h","m","r","g","z","d","b","p"]
hl=["w","j"]
parser=argparse.ArgumentParser(description="CharacterChecker")
parser.add_argument('input_file',help="input file path")
args=parser.parse_args()
sst=""
after=""
if len(args.input_file)!=0:
    path=args.input_file
    ss =" "
    ons=""
    ren=False
    print(path)
    if os.path.exists(path):
        f=open(path, "r" ,encoding='utf-8')
        #文字列の訂正
        ssss=f
        for b in ssss:
            ss+=(b)
        i=0
        f.close()
        que=[]
        while i<len(ss) :
            if (i+1)<len(ss) and word.__contains__(ss[i:i+2]):
                ons+=(word[ss[i:i+2]])
                sst+=ss[i:i+2]
                i+=2
                ren=False
            elif word.__contains__(ss[i]):
                ons+=(word[ss[i]])
                sst+=ss[i:i+2]
                i+=1
                ren=False
            else:
                if ren == b"\xef\xbb\xbf".decode("utf-8"):
                    pass
                elif ren:
                    que[-1][1]=i
                    i+=1
                else:
                    que.append([i,i])
                    i+=1
                    ren=True
        j=0
        ons_add=[" " for _ in range(len(que))]
        if len(que)!=0:
            print("不明な文字が%d見つかりました" % (len(que)))
            print("読み方を教えてください。※文字でなければ「パス」と入力")
            print("ひとつ前に戻るときは「戻る」と入力")
            while j < len(que)+1:
                if j==len(que):
                    print("以上です。見直し大丈夫ですか？OKで終了。")
                    ms=str(input())
                    if ms=="OK" :
                        break
                    else :
                        j=0
                i=que[j][0]
                m=que[j][1]
                print("\n---")
                k=""+ss[max(0,i-20):i]+"\""+ss[i:m+1]+"\""+ss[m+1:min(m+20,len(ss))]
                print(k)
                print(ss[i:m+1].encode('utf-8'))
                print("---\n")
                mm=str(input())
                if mm!="パス":
                    w=0
                    ons2=""
                    c=True
                    while w<len(mm):
                        if (w+1)<len(mm) and word.__contains__(mm[w:w+2]):
                            ons2+=(word[mm[w:w+2]])
                            sst+=mm[w:w+2]
                            w+=2
                        elif word.__contains__(mm[w]):
                            ons2+=(word[mm[w]])
                            sst+=mm[w:w+2]
                            w+=1
                        else:
                            print("不明な文字があります。やり直します。")
                            c=False
                            break
                    if c:
                        ons_add[j]=ons2
                        j+=1
                elif mm!="戻る":
                    if j!=0:
                        j-=1
                else:
                    ons+="?"
        cn=0
        for i in que:
            ons=ons[0:i]+ons_add[cn]+ons[i+1:-1]
            cn+=1
        ons+="?"
        print(ons)
        print("SUCCESS_文字列の音素変換に成功")
        #音素一覧表の作成
        ld=dict()
        for j in bo:
            for i in bo:
                bs=i+j
                ld[bs]=0
            for i in si:
                bs=j+i
                ld[bs]=0
                bs=j+i+"R"
                ld[bs]=0
                bs=i+j
                ld[bs]=0
                bs=i+"R"+j
                ld[bs]=0
            for i in hl:
                bs=j+i
                ld[bs]=0
                bs=i+j
                ld[bs]=0
            bs=j+"N"
            ld[bs]=0
            bs="N"+j
            ld[bs]=0
            bs=j+"Q"
            ld[bs]=0
            bs="Q"+j
            ld[bs]=0
            bs=j+"H"
            ld[bs]=0
            bs="H"+j
            ld[bs]=0
            bs=j+"?"
            ld[bs]=0
            bs="?"+j
            ld[bs]=0
        for i in si:
            bs="N"+i
            ld[bs]=0
            bs="Q"+i
            ld[bs]=0
            bs="?"+i
            ld[bs]=0
            bs="H"+i
            ld[bs]=0
        for i in hl:
            bs="N"+i
            ld[bs]=0
            bs="Q"+i
            ld[bs]=0
            bs="?"+i
            ld[bs]=0
        bs="N?"
        ld[bs]=0
        bs="?N"
        ld[bs]=0
        bs="Q?"
        ld[bs]=0
        bs="?Q"
        ld[bs]=0
        bs="fa"
        ld[bs]=0
        bs="fi"
        ld[bs]=0
        bs="fe"
        ld[bs]=0
        bs="fo"
        ld[bs]=0
        bs="af"
        ld[bs]=0
        bs="if"
        ld[bs]=0
        bs="uf"
        ld[bs]=0
        bs="ef"
        ld[bs]=0
        bs="of"
        ld[bs]=0
        bs="?f"
        ld[bs]=0
        bs="Qf"
        ld[bs]=0
        print("SUCCESS_２連音素配列の生成に成功")
        #文字列を音素に変換
        i=0
        while i<len(ons):
            if (i+3)<len(ons) and ld.__contains__(ons[i:i+3]):
                ld[ons[i:i+3]]+=1
                i+=1
            elif ld.__contains__(ons[i:i+2]):
                ld[ons[i:i+2]]+=1
                i+=1
            elif ons[i] =="?" and i==len(ons)-1:
                break
            elif ons[i:i+2]=="??":
                i+=1
            else:
                if ons[i:i+2].startswith("R"):
                    i+=1
                else:
                    print(ons[i:i+2])
                    exit(1)
        print("SUCCESS_計測完了")
        uu=[]
        uuu=[]
        used=[]
        lis=[]
        keys=[]
        i=0
        for i in ld.keys():
            if ld[i]!=0:
                used.append(i)
            else:
                if i.startswith("Q") and (i.endswith("a") or i.endswith("i") or i.endswith("u") or i.endswith("e") or i.endswith("o")) :
                    uuu.append(i)
                elif i.endswith("Hi"):
                    uuu.append(i)
                elif i.endswith("dH"):
                    uuu.append(i)
                elif i.startswith("dH"):
                    uuu.append(i)
                elif i==("di")or i==("ji") or i==("wu")or i==("gRe")or i==("mRe")or i==("rRe") or i==("pRe")or i==("kRe")or i==("bRe")or i==("je"):
                    uuu.append(i)
                elif i==("?Q") or i==("?N") or i==("Q?") or i==("Qn") or i==("Qw"):
                    uuu.append(i)
                else :
                    uu.append(i)
            lis.append(ld[i])
            keys.append(i)
        print("SUCCESS_カウント完了")
        #スタッツ表示
        print("使っていない音素列")
        print(uu)
        print("使用率")
        per=len(used)/(len(uu)+len(used))
        print("%f (%d/%d)" % (per,len(used),(len(uu)+len(used))))
        f=open("result.txt", "w")
        for r in ons:
            f.writelines(r)
        f.flush()
        f.close()
        ll=np.asarray(lis)
        ks=np.asarray(keys)
        print("%d(%d) sample:%s" % ( np.max(ll),len(ll[ll==np.max(lis)]),ks[ll==np.max(lis)]))

        print(np.mean(ll))
        plt.subplot(2,1,1)
        plt.hist(lis,bins=100)
        plt.subplot(2, 1, 2)
        plt.plot(ll)
        plt.show()
    else:
        print("ファイルが存在しません")