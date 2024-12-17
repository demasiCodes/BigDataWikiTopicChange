import requests
from datetime import datetime
import numpy as np

def get_revision_summary(page_title):
    url = "https://en.wikipedia.org/w/api.php"
    session = requests.Session()

    summary = {
        "page_title": page_title,
        "earliest_revision": None,
        "latest_revision": None
    }

    # Fetch the earliest revision
    params_earliest = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": page_title,
        "rvprop": "timestamp|size",
        "rvdir": "newer",
        "rvlimit": 1
    }
    response = session.get(url, params=params_earliest).json()
    pages = response.get("query", {}).get("pages", {})
    for page in pages.values():
        if "revisions" in page:
            rev = page["revisions"][0]
            summary["earliest_revision"] = {
                "timestamp": rev["timestamp"],
                "size": rev["size"]
            }

    # Fetch the latest revision
    params_latest = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": page_title,
        "rvprop": "timestamp|size",
        "rvdir": "older",
        "rvlimit": 1
    }
    response = session.get(url, params=params_latest).json()
    pages = response.get("query", {}).get("pages", {})
    for page in pages.values():
        if "revisions" in page:
            rev = page["revisions"][0]
            summary["latest_revision"] = {
                "timestamp": rev["timestamp"],
                "size": rev["size"]
            }

    return summary

# List of pages to summarize
pages = [
    "Feminism", "Gender studies", "Cancel culture", "Equal opportunity", 
    "Gender pay gap", "Intersectionality", "LGBTQ", "Glass ceiling", 
    "Transphobia", "Identity politics", "Racism in the United States"
]

all_pages = ["Human sexuality", "Gender studies", "Feminism", "LGBTQ", "LGBTQ rights in the United States", "Intersectionality", "Masculinity", "Queer", "Woman", "Gender role", 
         "Women's rights", "Conservatism", "Progressivism", "Left-wing politics", "Right-wing politics", "Stonewall riots", "Transgender rights movement", "Abortion debate", "Legal status of transgender people", "Black feminism", 
         "Critical race theory", "Identity politics", "Ethnicity", "Race (human categorization)", "Gender equality", "Media portrayal of LGBTQ people", "LGBTQ rights opposition", "Culture war", "Political polarization", "TERF (acronym)", 
         "Gender-critical feminism", "Radical feminism", "Global feminism", "LGBTQ rights in Africa", "First-wave feminism", "Second-wave feminism", "Don't ask, don't tell", "Obergefell v. Hodges", "Title IX", "Gender dysphoria", 
         "Sexual orientation", "Postmodern feminism", "Ecofeminism", "Liberal feminism", "Marxist feminism", "Queer theory", "Identity (social science)", "Asexuality", "Bisexuality", "Homophobia",
         "Non-binary gender", "Transphobia", "Pinkwashing (LGBTQ)", "Drag queen", "Conversion therapy", "Men's rights movement", "Toxic masculinity", "Misogyny", "Sexual harassment", "Feminization (sociology)",
         "Glass ceiling", "Gender pay gap", "Stereotype", "Sexual revolution", "Abortion", "Sex education", "Sexual and reproductive health", "Social conservatism", "Reproductive rights", "Rape culture",
         "Sexual objectification", "Stereotypes of Hispanic and Latino Americans in the United States", "Slut-shaming", "Victim blaming", "Third-wave feminism", "Fourth-wave feminism", "International Women's Day", "Violence against women", "Hate crime", "Discrimination",
         "Reverse discrimination", "Affirmative action", "Mattachine Society", "Daughters of Bilitis", "Harvey Milk", "ACT UP", "Compton's Cafeteria riot", "Same-sex marriage in the United States", "Pulse nightclub shooting", "MeToo movement",
         "African feminism", "Women's liberation movement in Asia", "Feminism in Latin America", "Islamic feminism", "LGBTQ rights in Europe", "LGBTQ rights in Asia", "LGBTQ rights in the Americas", "LGBTQ rights in the Middle East", "Convention on the Elimination of All Forms of Discrimination Against Women", "Sexism",
         "Occupational inequality", "World Conference on Women, 1995", "Women and the environment", "Women in government", "Women in the workforce", "Feminization of poverty", "Female education", "UN Women", "Women's empowerment", "Diversity, equity, and inclusion",
         "Disney and LGBTQ representation in animation", "Gender representation in video games", "Gender in advertising", "Lesbian literature", "Gay literature", "Feminist art movement", "New queer cinema", "Roe v. Wade", "Planned Parenthood v. Casey", "Equal Rights Amendment",
         "Violence Against Women Act", "National Organization for Women", "Human Rights Campaign", "Guttmacher Institute", "Lambda Legal", "Patriarchy", "Matriarchy", "YesAllWomen", "Feminist digital humanities", "Cyberbullying", 
         "Deplatforming", "White feminism", "Afro-pessimism (United States)", "Latinx", "Christian feminism", "Jewish feminism", "Religion and LGBTQ people", "Unpaid work", "United States v. Windsor", "Romer v. Evans", "Allophilia", "Amatonormativity", "Bias", "Cisnormativity", "Civil liberties", "Dehumanization", "Diversity (politics)", "Multiculturalism", "Neurodiversity", "Ethnic penalty",
         "Figleaf (linguistics)", "Gender-blind", "Heteronormativity", "History of eugenics", "Internalized oppression", "Masculism", "Medical model of disability", "Controversies in autism", "Net bias", "Oikophobia",
         "Oppression", "Police brutality", "Political correctness", "Polyculturalism", "Power distance", "Prejudice", "Prisoner abuse", "Racial bias in criminal news in the United States", "Racism by country", "Race relations", 
         "Racism", "International Convention on the Elimination of All Forms of Racial Discrimination", "Racism in Asia", "Racism in Africa", "Racism in Europe", "Racism in the Arab world", "Racial color blindness", "Religious intolerance", "Reverse racism", "Second-generation gender bias",
         "Snob", "Social exclusion", "Social identity threat", "Social model of disability", "Social privilege", "Christian privilege", "Male privilege", "White privilege", "Social stigma", "Speciesism",
         "Stereotype threat", "The talk (racism in the United States)", "Socioeconomic status", "Institutional discrimination", "Structural discrimination", "Statistical discrimination (economics)", "Taste-based discrimination", "Ageism", "Caste", "Class discrimination",
         "Dialect discrimination", "Defamation", "Ableism", "Genetic discrimination", "Discrimination based on hair texture", "Social justice", "Height discrimination", "Linguistic discrimination", "Lookism", "Sanism",
         "Discrimination based on skin tone", "Scientific racism", "Rankism", "Sexual orientation discrimination", "Sizeism", "Viewpoint discrimination", "Aromanticism", "Discrimination against asexual people", "Adultism", "Persecution of people with albinism",
         "Discrimination against autistic people", "Discrimination against homeless people", "Discrimination against drug addicts", "Anti-intellectualism", "Discrimination against intersex people", "Bias against left-handed people", "Anti-Masonry", "Aporophobia", "Audism", "Biphobia", 
         "Clan", "Elitism", "Ephebiphobia", "Social determinants of health", "Social determinants of health in poverty", "Social determinants of mental health", "Social stigma of obesity", "Discrimination against gay men", "Gerontophobia", "Heterosexism",
         "Discrimination against people with HIV/AIDS", "Leprosy stigma", "Discrimination against lesbians", "Discrimination against men", "Misandry", "Nepotism", "Fear of children", "Perpetual foreigner", "Pregnancy discrimination", "Employment discrimination",
         "Sectarianism", "Supremacism", "White supremacy", "21st-century anti-trans movement in the United Kingdom", "Gender Recognition Act 2004", "Gender nonconformity", "Gender expression", "Discrimination against non-binary people", "Transmisogyny", "Discrimination against transgender men",
         "Vegaphobia", "Xenophobia", "Anti-Afghan sentiment", "Anti-African sentiment", "Anti-Albanian sentiment", "Anti-Arab racism", "Anti-Armenian sentiment", "Racism against Asians", "Anti-Asian racism in France", "Racism in South Africa", 
         "Racism in the United States", "Anti-Assyrian sentiment", "Anti-Azerbaijani sentiment", "Anti-Black racism", "Racism against African Americans", "Racism in China", "Racism in South Africa", "Anti-Bengali sentiment", "Law for the Protection of Macedonian National Honour", "Anti-Catalan sentiment",
         "Anti-Chechen sentiment", "Anti-Chinese sentiment", "Anti-Colombian sentiment", "Anti-Croat sentiment", "Anti-Filipino sentiment", "Anti-Fulani sentiment", "Anti-Finnish sentiment", "Anti-Georgian sentiment", "Anti-Greek sentiment", "Anti-Haitian sentiment in the Dominican Republic",
         "Human Rights Watch", "Persecution of Hazaras", "Anti-Hungarian sentiment", "Anti-Igbo sentiment", "Anti-Indian sentiment", "Racism in Australia", "Racism in Canada", "Racism against Native Americans in the United States", "Anti-Irish sentiment", "Anti-Italianism",
         "Anti-Japanese sentiment", "Antisemitism", "New antisemitism", "Anti-Korean sentiment", "Anti-Kurdish sentiment", "Anti-Lithuanian sentiment", "Anti-Malay sentiment", "Anti-Māori sentiment", "Anti-Mexican sentiment", "Anti–Middle Eastern sentiment",
         "Anti-Mongolianism", "Anti-Nigerian sentiment", "Anti-Pakistan sentiment", "Anti-Palestinianism", "Anti-Pashtun sentiment", "Anti-Polish sentiment", "Anti-Quebec sentiment", "Anti-Romani sentiment", "Anti-Romanian sentiment", "Anti-Scottish sentiment",
         "Anti-Serb sentiment", "Anti-Slavic sentiment", "Anti-Somali sentiment", "Tatarophobia", "Anti-Thai sentiment", "Anti-Turkish sentiment", "Anti-Ukrainian sentiment", "Uyghurs", "List of incidents of xenophobia during the Venezuelan refugee crisis", "Anti-Vietnamese sentiment",
         "Cultural relationship between the Welsh and the English", "Discrimination against atheists", "Exclusivism", "Persecution of Baháʼís", "Persecution of Buddhists", "Anti-Christian sentiment", "Persecution of Christians", "Anti-Catholicism", "Persecution of Eastern Orthodox Christians", "Persecution of Jehovah's Witnesses",
         "Anti-Mormonism", "Persecution of Orthodox Tewahedo Christianity", "Persecution of Christians in the post–Cold War era", "Persecution of Falun Gong", "Anti-Hindu sentiment", "Persecution of Hindus", "Untouchability", "Islamophobia", "Persecution of Muslims", "Persecution of Ahmadis",
         "Anti-Shi'ism", "Persecution of Sufis", "Anti-Sunnism", "Persecution of minority Muslim groups", "Religious antisemitism", "Persecution of Jews", "Religious discrimination against modern pagans", "Anti-Protestantism", "Persecution of Rastafari", "Anti-Sikh sentiment",
         "Persecution of Yazidis", "Persecution of Zoroastrians", "Anti-LGBTQ rhetoric", "List of organizations designated by the Southern Poverty Law Center as anti-LGBTQ hate groups", "2020s anti-LGBTQ movement in the United States", "Anti-Zionism", "Blood libel", "Bullying", "Cancel culture", "Capital punishment for homosexuality",
         "Compulsory sterilization", "Corrective rape", "Counter-jihad", "Cultural genocide", "Democide", "Disability hate crime", "Dog whistle (politics)", "Domicide", "Economic discrimination", "Discrimination in education",
         "Eliminationism", "Enemy of the people", "Ethnic cleansing", "Ethnic conflict", "Ethnic hatred", "Ethnic joke", "Ethnocide", "Discrimination of excellence", "Equal opportunity", "Meritocracy",
         "Gender-based dress codes", "Cosmetics policy", "High heel policy", "Forced conversion", "Freak show", "Gay bashing", "Gendercide", "Genital modification and mutilation", "Violence against LGBTQ people", "Hate group",
         "Hate speech", "Patient dumping", "Housing discrimination", "Indian rolling", "Kill Haole Day", "Lavender Scare", "LGBTQ grooming conspiracy theory", "Mortgage discrimination", "Stop Murder Music", "Native American mascot controversy",
         "Occupational segregation", "Political repression", "Racialization", "Sex-selective abortion", "Violence against transgender people", "Victimisation", "White flight", "White genocide conspiracy theory", "Forced assimilation", "Witch hunt",
         "Age of candidacy", "Limpieza de sangre", "Blood quantum laws", "Crime of apartheid", "Disability", "Gerontocracy", "Gerrymandering", "Ghetto benches", "Internment", "Jewish quota", 
         "Law for Protection of the Nation", "Blood donation restrictions on men who have sex with men", "No kid zone", "Numerus clausus", "One-drop rule", "Racial quota", "Racial segregation", "Jim Crow laws", "Nuremberg Laws", "Racial steering",
         "Redlining", "Same-sex marriage", "Geographical segregation", "Age segregation", "Religious segregation", "Sex segregation", "Sodomy law", "State atheism", "State religion", "Ugly law",
         "Voter suppression", "Anti-discrimination law", "Anti-racism", "Constitutional colorblindness", "Cultural assimilation", "Cultural pluralism", "Diversity training", "Empowerment", "Fat acceptance movement", "Fighting Discrimination",
         "Hate speech laws by country", "Human rights", "Intersex human rights", "LGBTQ rights by country or territory", "Nonviolence", "Racial integration", "Reappropriation", "Self-determination", "Social integration", "Toleration"]

# Fetch summaries for all pages
summaries = []
for page in all_pages:
    summary = get_revision_summary(page)
    summaries.append(summary)

# Extended summary calculations
earliest_dates = [
    datetime.fromisoformat(s["earliest_revision"]["timestamp"][:-1]) 
    for s in summaries if s["earliest_revision"]
]
earliest_sizes = [s["earliest_revision"]["size"] for s in summaries if s["earliest_revision"]]
latest_dates = [
    datetime.fromisoformat(s["latest_revision"]["timestamp"][:-1]) 
    for s in summaries if s["latest_revision"]
]
latest_sizes = [s["latest_revision"]["size"] for s in summaries if s["latest_revision"]]

# Helper functions for mean and median of datetime objects
def datetime_mean(dates):
    return datetime.fromtimestamp(np.mean([dt.timestamp() for dt in dates]))

def datetime_median(dates):
    return datetime.fromtimestamp(np.median([dt.timestamp() for dt in dates]))

# Calculate statistics
extended_summary = {
    "average_earliest_date": {
        "mean": datetime_mean(earliest_dates).isoformat() if earliest_dates else None,
        "median": datetime_median(earliest_dates).isoformat() if earliest_dates else None
    },
    "average_earliest_size": {
        "mean": np.mean(earliest_sizes) if earliest_sizes else None,
        "median": np.median(earliest_sizes) if earliest_sizes else None
    },
    "earliest_earliest_revision": min(earliest_dates).isoformat() if earliest_dates else None,
    "latest_earliest_revision": max(earliest_dates).isoformat() if earliest_dates else None,
    "average_latest_date": {
        "mean": datetime_mean(latest_dates).isoformat() if latest_dates else None,
        "median": datetime_median(latest_dates).isoformat() if latest_dates else None
    },
    "average_latest_size": {
        "mean": np.mean(latest_sizes) if latest_sizes else None,
        "median": np.median(latest_sizes) if latest_sizes else None
    },
    "earliest_latest_revision": min(latest_dates).isoformat() if latest_dates else None,
    "latest_latest_revision": max(latest_dates).isoformat() if latest_dates else None,
}

# Print extended summary
print("\nExtended Summary:")
for key, value in extended_summary.items():
    print(f"{key}: {value}")