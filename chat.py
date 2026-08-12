# requirements: sentence-transformers, faiss-cpu, openai

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = OpenAI(
    base_url=azure_endpoint,
    api_key=azure_api_key
)

# Step 1: Prepare documents
documents = [
    # --- MEMBERSHIP & PRICING ---
    "Membership plans: Basic (3x/week) costs €79/month, Unlimited costs €109/month, Student Unlimited costs €79/month with valid student ID. Family discount: 20% off second member.",
    "Membership contracts run month-to-month. Cancel anytime with 30 days written notice via email to members@ironforge.fi.",
    "New member sign-up requires a mandatory 3-session OnRamp course (€60 one-time fee) before joining regular classes. OnRamp teaches foundational movements.",
    "Drop-in fee for non-members is €25 per class. Visiting CrossFit athletes from other affiliates get a discounted drop-in rate of €18 with proof of membership.",
    "Membership freeze is available up to 2 months per year (e.g. vacation, injury). Freeze requests must be submitted 7 days in advance. Frozen months are not billed.",

    # --- CLASS SCHEDULE ---
    "Weekday class times: 06:00, 07:00, 09:00, 12:00, 16:00, 17:00, 18:00, 19:00. Saturday: 09:00, 10:00, 11:00. Sunday: 10:00 open gym only.",
    "Class booking opens 48 hours in advance through the IronForge app or website. Cancellation must be done at least 2 hours before class to avoid a no-show fee of €5.",
    "Maximum class size is 16 athletes. Waitlist is available — if a spot opens within 12 hours of class, waitlisted members are automatically booked and notified via app.",
    "Open Gym is available Monday–Friday 06:00–21:00 and weekends 09:00–14:00 for Unlimited members only. Basic members must attend coached classes.",
    "Special programming: Every Saturday 10:00 class is a Partner WOD. First Sunday of each month is a free community workout open to the public at 11:00.",

    # --- COACHING & TRAINERS ---
    "Head Coach Mikael Virtanen holds CrossFit Level 3 certification and has 9 years of coaching experience. He leads the 06:00 and 17:00 weekday classes.",
    "Coach Sara Leinonen specializes in Olympic weightlifting and mobility. She runs the Wednesday 18:00 Barbell Club session (open to all Unlimited members).",
    "Coach Joonas Mäkinen is a CrossFit Level 2 coach and certified nutritionist. He offers personal training sessions at €70/hour and monthly nutrition consultations.",
    "Personal training sessions can be booked in packs: 5 sessions €320, 10 sessions €600. Sessions expire 3 months after purchase.",
    "All IronForge coaches hold valid first aid certification. An AED defibrillator is located at the front desk and near the lifting platform.",

    # --- PROGRAMMING & WORKOUTS ---
    "IronForge follows a 5-day cycle programming (Mon–Fri) with structured strength work followed by a MetCon (metabolic conditioning WOD). Saturdays are skill + partner WODs.",
    "Strength focus rotates every 8 weeks: current cycle (Q3 2025) focuses on Back Squat and Strict Press. Previous cycle focused on Deadlift and Bench Press.",
    "All workouts are scaled to three levels: Rx (as prescribed), Intermediate, and Beginner. Coaches provide scaling options at the start of every class.",
    "Benchmark WODs (e.g. Fran, Grace, Murph) are programmed once per quarter so athletes can track progress. Results are logged in the IronForge app.",
    "IronForge participates in the CrossFit Open every year (February–March). Box-wide leaderboard and Friday Night Lights events are organized during the Open.",

    # --- NUTRITION ---
    "IronForge follows general CrossFit nutrition principles: eat whole foods, avoid processed sugar, balance protein, carbs, and fat based on your training load.",
    "Coach Joonas offers a Nutrition Kickstart Package: initial consultation + 4-week meal plan + one follow-up session for €120.",
    "The box sells protein supplements (whey and plant-based), creatine, and magnesium at the front desk. Brand: Nordic Whey. Whey protein: €45/kg, plant-based: €49/kg.",
    "Post-workout nutrition recommendation: consume 20–40g protein and fast-digesting carbs within 30–45 minutes after training for optimal recovery.",
    "IronForge does not prescribe specific diets, but coaches are familiar with Paleo, Zone, and macro-tracking approaches and can advise accordingly.",

    # --- EQUIPMENT & FACILITY ---
    "The box is equipped with 14 Rogue barbell stations, GHD machines, assault bikes, ski ergs, rowing machines (Concept2), and a 20m turf track for sled work.",
    "Equipment available: pull-up rigs, pegboards, rings, kettlebells (8kg–48kg), dumbbells, plyo boxes, wall balls, and a dedicated Olympic lifting platform.",
    "Locker rooms include showers. Lockers are available for day use (bring your own lock) or rented monthly for €10/month. Towels available at front desk for €2.",
    "Parking is free in the lot adjacent to the building. Bicycle racks and a repair station are available outside the main entrance.",
    "The box address is Teollisuuskatu 14, 00510 Helsinki. Contact: info@ironforge.fi | Tel: +358 9 123 4567. Instagram & Facebook: @ironforgecrossfit.",

    # --- SAFETY & INJURY ---
    "Injury or pain during training: stop immediately and inform the coach. Coaches are trained to assess and will recommend rest or referral to a physiotherapist.",
    "IronForge has a partnership with Physio Helsinki — members receive a 15% discount on physiotherapy sessions. Referral cards available at the front desk.",
    "New members must complete a health questionnaire before their first session. Members with prior injuries should inform their coach before each class.",
    "Movement standards are enforced for safety, not to be strict — coaches will correct form before adding load. 'Mechanics, then consistency, then intensity.'",

    # --- EVENTS & COMMUNITY ---
    "IronForge hosts an internal competition every June called the IronForge Throwdown — free to enter for members, spectators welcome.",
    "Monthly social events: quarterly team dinners, post-Murph BBQ on Memorial Day, and a Christmas party in December.",
    "IronForge supports local charity events. In 2024, the box raised €3,200 for children's sports programs through a 24-hour row fundraiser.",
    "Member referral program: refer a friend who signs up for a full membership and receive one free month added to your account.",

    # --- POLICIES ---
    "Gym etiquette rules: always clean equipment after use, return weights to racks, chalk stays in the chalk bucket, no phones on the gym floor during class.",
    "Children under 16 are not permitted on the gym floor during regular classes. A teens CrossFit program (ages 13–15) runs Tuesdays and Thursdays at 16:00.",
    "Music policy: coaches control the playlist during class. Members may suggest songs via the IronForge app's music request feature.",
    "Lost and found items are kept at the front desk for 2 weeks before being donated to charity.",
]

# Step 2: Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
embeddings = model.encode(documents)

# Step 3: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 4: Retrieval function
def retrieve(query, k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Step 5: RAG function
def rag_query(question, use_full_context=True, k=2):
    # Use all documents as context by default; retrieval remains optional.
    if use_full_context:
        context_docs = documents
    else:
        context_docs = retrieve(question, k=k)

    context = "\n".join(context_docs)

    # Create prompt
    prompt = f"""Answer the question based only on this context:
    Context:
    {context}
    Question: {question}
    Answer:"""

    # Generate response
    response = client.responses.create(
        # Azure OpenAI expects deployment name in model field.
        model=azure_deployment,
        input=prompt
    )

    return response.output_text or ""

# Test it
print(rag_query("Who coaches the Wednesday barbell session?"))
# Output: "An Unlimited membership costs €120 per month."