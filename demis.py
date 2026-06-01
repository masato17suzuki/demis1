"""
Demis Layer1 — 雇用主エンジン
スライダー × この水準で最も輝いた人材を生成 × 履歴書ダウンロード
"""
import numpy as np
import json
import io
import os

# streamlit/requests はUI（main・call_api）専用。gen_population は numpy のみ
# 使用するため、これらが無い実行環境でもL1モジュールを読めるよう import を
# ガードする。L1のUIロジック・雇用主要求の数式（threshold/SIGMA等）は一切
# 変更しない。L0とは別実装（年齢生成等が異なる）であり統合はしない。
try:
    import streamlit as st
    import requests
    _UI_OK = True
except ImportError:
    st = None
    requests = None
    _UI_OK = False

# ─────────────────────────────────────────
# 日本統計ベース給与
# ─────────────────────────────────────────

INDUSTRY_SALARY = {
    "IT/Web":480000,"製造":360000,"医療":390000,"金融":500000,
    "小売":290000,"公務":370000,"建設":380000,"物流":310000,"その他":330000,
}

# ─────────────────────────────────────────
# スライダーラベル
# ─────────────────────────────────────────

IQ_L  = ["極低(60台)","低め(70台)","やや低(80台)","平均以下(85-90)","平均(95-105)",
         "平均以上(105-110)","やや高(110-115)","高い(115-120)","非常に高い(120-130)","天才級(130+)"]
EDU_L = ["不問(中卒可)","高卒以上","高卒~専門","専門卒程度","専門~大卒",
         "大卒程度","大卒以上","有名大卒","難関大卒","旧帝・早慶以上"]
CC_L  = ["文化資本ゼロ","非常に低い","低い","やや低い","中程度",
         "やや高い","高い","非常に高い","極めて高い","最高水準"]
DCC_L = ["向上心なし","ほぼなし","低い","やや低い","標準的",
         "やや高い","高い","非常に高い","極めて高い","狂気的な向上心"]
FIT_L = ["体力ほぼゼロ","非常に低い","低い","やや低い","標準",
         "やや高い","高い","非常に高い","アスリート級","超人的体力"]

# ─────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────

def cl(v,a,b): return max(a,min(b,v))
def sig(z): return 1.0/(1.0+np.exp(-np.clip(z,-500,500)))
def sp(x,th,sg,k=3.0): return sig(k*(x-th)/sg)

def threshold(s, dim):
    t=(s-1)/9
    if dim=="iq":  return 60+t*70
    if dim=="edu": return t*1.0
    if dim=="cc":  return 0.05+t*0.65
    if dim=="dcc": return t*0.20
    return 0.05+t*0.80

SIGMA={"iq":15,"edu":0.28,"cc":0.15,"dcc":0.04,"fit":0.13}

def calc_market_wage(industry, s_iq, s_edu, s_cc, s_dcc, s_fit):
    base=INDUSTRY_SALARY.get(industry,330000)
    m=1.0+(s_iq-5)*0.04+(s_edu-5)*0.05+(s_cc-5)*0.02+(s_dcc-5)*0.02+(s_fit-5)*0.01
    return int(base*max(0.5,m))

# ─────────────────────────────────────────
# 人口生成（マッチング用）
# ─────────────────────────────────────────

def gen_population(n=1000, mode="jp", seed=None):
    rng=np.random.default_rng(seed)
    people=[]
    EDU=[("高学歴",1.0),("大卒",0.65),("専門",0.38),("高卒",0.18),("中卒",0.05)]
    IND=["IT","製造","医療","金融","小売","公務","その他"]
    for _ in range(n):
        g="男" if rng.random()<0.5 else "女"
        if mode=="jp":
            a=int(cl(round(rng.normal(38,12)),18,65))
            iq=int(cl(round(100+rng.normal(0,15)),60,160))
            b=rng.random()
            bst=cl((iq-100)/15,-1,2)
            pH=cl(0.08+bst*0.04,0.01,0.30)
            pU=cl(0.30+bst*0.05,0.05,0.50)
            if b<pH:           edu,es="高学歴",1.0
            elif b<pH+pU:      edu,es="大卒",0.65
            elif b<pH+pU+0.25: edu,es="専門",0.38
            elif b<pH+pU+0.57: edu,es="高卒",0.18
            else:              edu,es="中卒",0.05
            ind=IND[int(rng.choice(len(IND),p=[0.20,0.18,0.15,0.12,0.12,0.08,0.15]))]
            cc=cl(0.45+(iq-100)/150+rng.normal(0,0.15),0,1)
            dcc=cl(0.04+rng.normal(0,0.04),0,0.25)
            bm={(18,29):0.80,(30,39):0.70,(40,49):0.60,(50,59):0.45,(60,99):0.30}
            bf={(18,29):0.60,(30,39):0.50,(40,49):0.40,(50,59):0.30,(60,99):0.20}
            t=bm if g=="男" else bf
            base=next(v for (lo,hi),v in t.items() if lo<=a<=hi)
            fit=cl(base+rng.normal(0,0.08),0.05,1.0)
        else:
            a=int(rng.integers(18,65))
            iq=int(rng.integers(60,161))
            edu,es=EDU[int(rng.integers(0,5))]
            ind=IND[int(rng.integers(0,7))]
            cc=float(rng.random())
            dcc=float(rng.random()*0.25)
            fit=float(rng.random())
        ki=cl(dcc*3+rng.normal(0,0.12)+0.30,0,1)
        ika=cl((1-cc)*0.5+rng.normal(0,0.10)+0.15,0,1)
        ai=cl((a-25)/60+rng.normal(0,0.10)+0.15,0,1)
        raku=cl((iq-70)/90*0.6+rng.normal(0,0.12)+0.20,0,1)
        love=cl(fit*0.3+cc*0.3+rng.normal(0,0.12)+0.20,0,1)
        people.append({"gender":g,"age":a,"iq":iq,"edu":edu,"edu_score":es,
                       "industry":ind,"cc":cc,"dcc":dcc,"fitness":fit,
                       "ki":ki,"ika":ika,"ai":ai,"raku":raku,"love":love})
    return people

def calc_score(p, s_iq, s_edu, s_cc, s_dcc, s_fit):
    return (sp(p["iq"],       threshold(s_iq,"iq"),  SIGMA["iq"])  *
            sp(p["edu_score"],threshold(s_edu,"edu"),SIGMA["edu"]) *
            sp(p["cc"],       threshold(s_cc,"cc"),  SIGMA["cc"])  *
            sp(p["dcc"],      threshold(s_dcc,"dcc"),SIGMA["dcc"]) *
            sp(p["fitness"],  threshold(s_fit,"fit"),SIGMA["fit"]))

# ─────────────────────────────────────────
# API呼び出し
# ─────────────────────────────────────────

def call_api(api_key, prompt, provider, az_ep, az_dp):
    try:
        if provider=="Azure OpenAI" and az_ep and api_key:
            if "services.ai.azure.com" in az_ep:
                # Azure AI Foundry (services.ai.azure.com) の正しい形式
                url=az_ep.rstrip("/")+"/openai/deployments/"+az_dp+"/chat/completions?api-version=2024-10-21"
            else:
                url=az_ep.rstrip("/")+"/openai/deployments/"+az_dp+"/chat/completions?api-version=2025-01-01-preview"
            res=requests.post(url,
                headers={"api-key":api_key,"Content-Type":"application/json"},
                json={"max_completion_tokens":6000,"messages":[{"role":"user","content":prompt}]},
                timeout=30)
            data=res.json()
            if data.get("choices"): return data["choices"][0]["message"]["content"]
            # エラー詳細を返す
            return "APIERR:"+str(data)
        else:
            res=requests.post("https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization":"Bearer "+api_key,"Content-Type":"application/json"},
                json={"model":"deepseek-chat","max_tokens":900,"messages":[{"role":"user","content":prompt}]},
                timeout=30)
            data=res.json()
            if data.get("choices"): return data["choices"][0]["message"]["content"]
            return "APIERR:"+str(data)
    except Exception as ex:
        return "APIERR:"+str(ex)

# ─────────────────────────────────────────
# 「この水準で最も輝いた人材」を生成
# ─────────────────────────────────────────

def gen_shining_candidates(api_key, provider, az_ep, az_dp,
                            s_iq, s_edu, s_cc, s_dcc, s_fit,
                            industry, wage, hire_count, free_text, n=5):
    prompt = "\n".join([
        "あなたはDemis採用AIです。",
        "以下の水準に正直に合致する人材を"+str(n)+"人生成してください。",
        "絶対に美化・理想化しないでください。水準通りのリアルな人物を描いてください。",
        "水準1-3は低スペックでよい。その中で「その人なりに頑張った」エピソードを入れてください。",
        "例：知的水準2→中学卒業後すぐ働き始めた。字は読めるが計算は苦手。",
        "例：体力水準1→車椅子使用。手先は器用で在宅作業なら問題なし。",
        "",
        "【求人条件】",
        "業種："+industry,
        "月給："+str(wage)+"円",
        "採用人数："+str(hire_count)+"人",
        "雇用主の思い："+free_text,
        "",
        "【水準設定】",
        "IQ水準："+IQ_L[s_iq-1]+"（スライダー"+str(s_iq)+"/10）",
        "学歴："+EDU_L[s_edu-1]+"（スライダー"+str(s_edu)+"/10）",
        "文化資本："+CC_L[s_cc-1]+"（スライダー"+str(s_cc)+"/10）",
        "熱意・向上心："+DCC_L[s_dcc-1]+"（スライダー"+str(s_dcc)+"/10）",
        "体力："+FIT_L[s_fit-1]+"（スライダー"+str(s_fit)+"/10）",
        "",
        "以下のJSON形式のみで返答してください：",
        '{"candidates": [',
        '  {"name": "姓名（架空）", "age": 年齢, "gender": "男/女",',
        '   "headline": "この人物を一言で表すキャッチコピー",',
        '   "story": "この水準の中で輝いた理由・エピソード（2文）",',
        '   "strength": "最大の強み（1文）",',
        '   "honne": "本音・本当の動機（1文）",',
        '   "emotion": "喜怒哀楽愛の特徴（1文）"',
        '  }',
        ']}',
    ])
    result = call_api(api_key, prompt, provider, az_ep, az_dp)
    # APERRの場合は空リストで返す（呼び出し元でresultを確認）
    if not result or result.startswith("APIERR:"):
        return ["__ERR__:"+result]
    try:
        start=result.find("{"); end=result.rfind("}")+1
        parsed=json.loads(result[start:end])
        candidates=parsed.get("candidates",[])
        if candidates: return candidates
    except: pass
    try:
        start=result.find("["); end=result.rfind("]")+1
        return json.loads(result[start:end])
    except:
        return ["__ERR__:JSONパース失敗:"+result[:200]]


def build_resume(c, industry, wage, s_iq, s_edu, s_cc, s_dcc, s_fit):
    return "\n".join([
        "═"*44,
        "Demis 簡易履歴書",
        "═"*44,
        f"氏名：{c.get('name','—')}　{c.get('age','—')}歳・{c.get('gender','—')}性",
        f"キャッチコピー：{c.get('headline','—')}",
        "",
        "【Demis水準】",
        f"IQ：{IQ_L[s_iq-1]}　学歴：{EDU_L[s_edu-1]}",
        f"文化資本：{CC_L[s_cc-1]}　熱意：{DCC_L[s_dcc-1]}　体力：{FIT_L[s_fit-1]}",
        "",
        "【この水準で輝いた理由】",
        c.get('story','—'),
        "",
        "【最大の強み】",
        c.get('strength','—'),
        "",
        "【感情プロファイル】",
        c.get('emotion','—'),
        "",
        "【採用担当者参照：本音】",
        c.get('honne','—'),
        "",
        f"想定月給：{wage:,}円（業種：{industry}）",
        "═"*44,
    ])

# ─────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────

def main():
    st.set_page_config(page_title="Demis Layer1", layout="wide", page_icon="🏢")
    st.title("🏢 Demis Layer1 — 雇用主エンジン")
    st.caption("水準を設定 → その水準で最も輝いた人材を生成 → 履歴書ダウンロード")

    for k in ["candidates","population","unsatisfied_jobs"]:
        if k not in st.session_state:
            st.session_state[k] = None if k!="unsatisfied_jobs" else []

    # ─── サイドバー
    with st.sidebar:
        st.header("API設定")
        provider=st.radio("プロバイダー",["DeepSeek","Azure OpenAI"],horizontal=True,index=1)
        az_key=""; az_endpoint=""; az_deploy="gpt-5.1"
        if provider=="DeepSeek":
            api_key=st.text_input("DeepSeek APIキー",type="password",
                                  value=os.environ.get("DEEPSEEK_API_KEY",""))
        else:
            az_key     =st.text_input("Azure OpenAI APIキー",type="password",
                                      value=os.environ.get("AZURE_OPENAI_KEY",""))
            az_endpoint=st.text_input("エンドポイント",
                                      value=os.environ.get("AZURE_OPENAI_ENDPOINT",""))
            az_deploy  =st.text_input("デプロイ名",value="gpt-5.1")
            api_key=az_key
        st.session_state["provider"]    =provider
        st.session_state["az_endpoint"] =az_endpoint
        st.session_state["az_deploy"]   =az_deploy

        st.divider()
        st.caption("📤 求人をL0に送る")
        if st.button("この求人をL0に転送", use_container_width=True):
            job_packet = {
                "from": "Layer1",
                "industry": st.session_state.get("l1_industry","未設定"),
                "wage": st.session_state.get("l1_wage",300000),
                "hire_count": st.session_state.get("l1_hire",3),
                "requirements": {
                    "iq":  st.session_state.get("l1_iq",5),
                    "edu": st.session_state.get("l1_edu",5),
                    "cc":  st.session_state.get("l1_cc",5),
                    "dcc": st.session_state.get("l1_dcc",5),
                    "fit": st.session_state.get("l1_fit",5),
                },
                "free_text": st.session_state.get("l1_free",""),
            }
            st.download_button(
                "📥 JSONをダウンロード（L0にアップ）",
                json.dumps(job_packet, ensure_ascii=False, indent=2).encode("utf-8"),
                "job_from_L1.json",
                "application/json",
                use_container_width=True
            )

        st.divider()
        # 1000人データ
        if not st.session_state.population:
            gc1,gc2=st.columns(2)
            with gc1:
                if st.button("🇯🇵 日本統計",use_container_width=True):
                    with st.spinner("生成中..."):
                        st.session_state.population=gen_population(1000,"jp")
                    st.rerun()
            with gc2:
                if st.button("🎲 ランダム",use_container_width=True):
                    with st.spinner("生成中..."):
                        st.session_state.population=gen_population(1000,"rnd")
                    st.rerun()
        else:
            pop=st.session_state.population
            st.metric("母集団",f"{len(pop)}人")

    # ─── 雇用条件
    st.subheader("🏢 雇用条件")
    ec1,ec2=st.columns(2)
    with ec1:
        industry=st.selectbox("業種",list(INDUSTRY_SALARY.keys()))
        offer_wage=st.number_input("提示月給（円）",150000,2000000,
                                   INDUSTRY_SALARY.get(industry,330000),10000)
        hire_count=st.number_input("採用予定人数",1,100,3)
    with ec2:
        free_text=st.text_area("求める人材像・会社の思い",
            placeholder="例：熱意があれば学歴不問。体育会系文化。長く働いてほしい...",height=100)

    # ─── スライダー
    st.subheader("📊 求める人材の水準")
    sc1,sc2,sc3,sc4=st.columns([2,2,2,1])
    with sc1:
        s_iq =st.slider("知的水準",  1,10,5,key="l1_iq")
        s_edu=st.slider("学歴水準",  1,10,5,key="l1_edu")
    with sc2:
        s_cc =st.slider("文化資本",  1,10,5,key="l1_cc")
        s_dcc=st.slider("熱意・向上心",1,10,5,key="l1_dcc")
    with sc3:
        s_fit=st.slider("体力水準",  1,10,5,key="l1_fit")
        # 市場賃金
        market_wage=calc_market_wage(industry,s_iq,s_edu,s_cc,s_dcc,s_fit)
        gap=offer_wage-market_wage
        st.metric("市場適正月給",f"{market_wage:,}円",
                  delta=f"{'+' if gap>=0 else ''}{gap:,}円")
    with sc4:
        st.write("")
        st.write("")
        st.write("")
        gen_btn=st.button("✨ この水準で\n人材を求める",use_container_width=True)

    # 合致人数リアルタイム表示
    if st.session_state.population:
        pop=st.session_state.population
        total=sum(calc_score(p,s_iq,s_edu,s_cc,s_dcc,s_fit) for p in pop)
        st.metric("母集団合致人数",f"{round(total)}人")

    if gen_btn:
        if not api_key:
            st.warning("サイドバーにAPIキーを入力してください")
        else:
            with st.spinner("この水準で最も輝いた人材を探しています..."):
                candidates=gen_shining_candidates(
                    api_key,provider,az_endpoint,az_deploy,
                    s_iq,s_edu,s_cc,s_dcc,s_fit,
                    industry,offer_wage,hire_count,free_text or "特になし", n=5
                )
            if candidates and isinstance(candidates[0], str) and candidates[0].startswith("__ERR__:"):
                st.error("APIエラー："+candidates[0].replace("__ERR__:",""))
            elif candidates:
                st.session_state.candidates = candidates   # 新規5人でリセット
            else:
                st.error("候補者の生成に失敗しました（空の応答）")

    # 次の5人ボタン
    next_btn = False
    if st.session_state.candidates:
        _, nb_col = st.columns([3,1])
        with nb_col:
            next_btn = st.button("➕ 次の5人", use_container_width=True)

    if next_btn:
        if not api_key:
            st.warning("サイドバーにAPIキーを入力してください")
        else:
            with st.spinner("次の5人を探しています..."):
                more=gen_shining_candidates(
                    api_key,provider,az_endpoint,az_deploy,
                    s_iq,s_edu,s_cc,s_dcc,s_fit,
                    industry,offer_wage,hire_count,free_text or "特になし", n=5
                )
            if more and isinstance(more[0], str) and more[0].startswith("__ERR__:"):
                st.error("APIエラー："+more[0].replace("__ERR__:",""))
            elif more:
                st.session_state.candidates = st.session_state.candidates + more

    # ─── 候補者表示（5人ずつ追加）
    if st.session_state.candidates:
        st.divider()
        st.subheader(f"👥 候補者（{len(st.session_state.candidates)}人）")

        for i,c in enumerate(st.session_state.candidates):
            col_info, col_dl = st.columns([4,1])
            with col_info:
                st.markdown(
                    f"**{i+1}. {c.get('name','—')}**　"
                    f"{c.get('age','—')}歳・{c.get('gender','—')}性　"
                    f"「{c.get('headline','—')}」"
                )
                st.caption(c.get('story','—'))
                st.caption(f"強み：{c.get('strength','—')}　／　感情：{c.get('emotion','—')}")
            with col_dl:
                resume=build_resume(c,industry,offer_wage,s_iq,s_edu,s_cc,s_dcc,s_fit)
                st.download_button(
                    "📄 履歴書",
                    resume.encode("utf-8"),
                    f"resume_{i+1}_{c.get('name','candidate')}.txt",
                    "text/plain",
                    key=f"dl_{i}"
                )
            st.divider()

    # ─── 求人票作成
    st.divider()
    btn_l, btn_r = st.columns(2)
    with btn_l:
        if st.button("📝 AIアドバイスを元に求人票作成", use_container_width=True, key="job_ai"):
            if not api_key:
                st.warning("サイドバーにAPIキーを入力してください")
            else:
                prompt_job = "\n".join([
                    "以下の条件で求人票を作成してください。",
                    f"業種：{industry}　提示年収：{offer_wage:,}円　採用人数：{hire_count}人",
                    f"IQ水準：{s_iq}/10　学歴：{s_edu}/10　文化資本：{s_cc}/10",
                    f"熱意：{s_dcc}/10　体力：{s_fit}/10",
                    f"特記：{free_text or 'なし'}",
                    "",
                    "求人票として適切な表現に整え、応募者を惹きつける文面にしてください。",
                    "採用AIの分析に基づき、市場の人材供給状況を考慮した現実的な要件にしてください。",
                ])
                with st.spinner("AIが求人票を作成中..."):
                    result = call_api(api_key, prompt_job, provider, az_endpoint, az_deploy)
                st.session_state["job_draft"] = result
    with btn_r:
        if st.button("✍ AIアドバイスを無視して求人票作成", use_container_width=True, key="job_manual"):
            if not api_key:
                st.warning("サイドバーにAPIキーを入力してください")
            else:
                prompt_job = "\n".join([
                    "以下の条件をそのまま求人票にしてください。条件の緩和や変更はしないでください。",
                    f"業種：{industry}　提示年収：{offer_wage:,}円　採用人数：{hire_count}人",
                    f"IQ水準：{s_iq}/10　学歴：{s_edu}/10　文化資本：{s_cc}/10",
                    f"熱意：{s_dcc}/10　体力：{s_fit}/10",
                    f"特記：{free_text or 'なし'}",
                    "",
                    "雇用主の意向をそのまま反映し、一切の修正提案を加えないでください。",
                ])
                with st.spinner("求人票を作成中..."):
                    result = call_api(api_key, prompt_job, provider, az_endpoint, az_deploy)
                st.session_state["job_draft"] = result

    if st.session_state.get("job_draft"):
        st.code(st.session_state["job_draft"], language=None)
        st.download_button("📄 求人票をダウンロード",
            st.session_state["job_draft"].encode("utf-8"),
            "demis_job_posting.txt", "text/plain", key="dl_job")

    # ─── シミュレーションモード
    st.divider()
    with st.expander("📊 シミュレーションモード"):
        st.caption("求人1000件をJSON生成してE層に転送")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("🇯🇵 日本型統計", use_container_width=True, key="sim_jp"):
                with st.spinner("1000件生成中..."):
                    pop = gen_population(1000, "jp")
                    data = [dict(p) for p in pop]
                    st.session_state["sim_data"] = json.dumps(data, ensure_ascii=False)
                st.success("日本型1000件生成完了")
        with sc2:
            if st.button("🎲 ランダム", use_container_width=True, key="sim_rnd"):
                with st.spinner("1000件生成中..."):
                    pop = gen_population(1000, "rnd")
                    data = [dict(p) for p in pop]
                    st.session_state["sim_data"] = json.dumps(data, ensure_ascii=False)
                st.success("ランダム1000件生成完了")
        with sc3:
            if st.button("📊 今回のレベル", use_container_width=True, key="sim_cur"):
                with st.spinner("1000件生成中..."):
                    pop = gen_population(1000, "jp")
                    data = [dict(p) for p in pop]
                    st.session_state["sim_data"] = json.dumps(data, ensure_ascii=False)
                st.success("現在レベル1000件生成完了")

        if st.session_state.get("sim_data"):
            st.download_button("📥 1000件JSONをダウンロード",
                st.session_state["sim_data"].encode("utf-8"),
                "demis_1000jobs.json", "application/json",
                use_container_width=True, key="dl_sim")

if __name__=="__main__":
    main()
