# TODO: provare a generare domande in lingue diverse dalla lingua del contesto/documento
## Separare le due versioni ITA/ENG senza modificare la lingua del contesto
QUESTION_GEN_PROMPT = """
You are a cyber analyst specialist with many years of experience. Your task is to create synthetic questions focused on cyber security field from a series of <CHUNK> taken from a <DOCUMENT> (retrieved from a <URL>). Given a <CHUNK> of context about some topic(s), generate {number_of_questions} examples of <QUESTION> a user could ask and would be answered using information from the <CHUNK>. To perform the task follow these rules:
    - the <QUESTION> MUST always be in the same language as the content of the <CHUNK>.
    - DO NOT EVER REFER to the things like  "the document", "the text", "the regulation" in the <QUESTION>, if you want to refer to the <DOCUMENT> use the appropriate <DOCUMENT> name if stated in the <CHUNK>.
    - the <QUESTION> should be technical on the cybersecurity field
    - use only the information inside the <CHUNK> to create the <QUESTIONs>
    - make a list of <QUESTIONs>, each <QUESTION> divided by a new line
    
    
here a summary of the information:
    - <DOCUMENT> a document regarding cybersecurity
    - <URL> a link to the <DOCUMENT>
    - <CHUNK> a chunk from the <DOCUMENT>
    - <QUETION> the question you must create from the <CHUNK>

example of list of <QUESTIONs>:
<QUESTION1>
<QUESTION2>
<QUESTION3>

"""

# - generate the <QUESTION> only if the chunk contains information about cyber security or anything related to cyber security and computer science. If it does not include any information just mark it as <NOQUESTION>
QUESTION_GEN_QUESTION_HUMAN_MESSAGE = """The <CHUNK> comes from the following <URL>: {url}.\nGenerate {number_of_questions} <QUETIONs> in {language} from the following <CHUNK> of text following the rules that have been explained before:\n\n{chunk}"""

QUESTION_GEN_QUESTION_EXAMPLE_1 = "Data Management (DP-ID.DM): i dati personali sono trattati attraverso processi definiti, in coerenza con le normative di riferimento. DP-ID.DM-1: Il ciclo di vita dei dati è definito e documentato. DP-ID.DM-2: Sono definiti, implementati e documentati i processi riguardanti l'informazione dell'interessato in merito al trattamento dei dati. DP-ID.DM-3: Sono definiti, implementati e documentati i processi di raccolta e revoca del consenso dell'interessato al trattamento di dati. DP-ID.DM-4: Sono definiti, implementati e documentati i processi per l'esercizio dei diritti (accesso, rettifica, cancellazione, ecc.) dell'interessato. DP-ID.DM-5: Sono definiti, implementati e documentati i processi di trasferimento dei dati in ambito internazionale."
QUESTION_GEN_QUESTION_EXAMPLE_COMPILED_1 = QUESTION_GEN_QUESTION_HUMAN_MESSAGE.format(
    url="https://www.acn.gov.it/portale/strategia-nazionale-di-cybersicurezza",
    number_of_questions=1,
    language="italian",
    chunk=QUESTION_GEN_QUESTION_EXAMPLE_1,
)
QUESTION_GEN_ANSWER_EXAMPLE_1 = "Quali sono i principali requisiti riguardanti la gestione dei dati personali secondo il framework Framework Nazionale per la Cybersecurity e la Data Protection?"


ANSWER_GEN_PROMPT = """

Answer the <QUESTION> using the information given in the <CONTEXT>. To perform the task follow rules explained before: 
- The answer should be exhaustive and should use the same terminology presented in the <CONTEXT>.
- The answer should match the same language used in the chunk (<CONTEXT> in english needs an answer in english, <CONTEXT> in italian needs an answer in italian).\n

<CONTEXT>: {context}\n
<QUESTION>: {question}
"""

ANSWER_GEN_SYSTEM_PROMPT = """You are a cyber analyst specialist with many years of experience. Your job is to answer a <QUESTION> following this set of rules:
- The answer should be exhaustive and should use the same terminology presented in the <CONTEXT>.
- The answer should match the same language used in the chunk (<CONTEXT> in english needs an answer in english, <CONTEXT> in italian needs an answer in italian)"""

ANSWER_GEN_QUESTION_EXAMPLE_1 = "How does the NIST Cybersecurity Framework 2.0 address the integration of cybersecurity and enterprise risk management (ERM)?"

ANSWER_GEN_ANSWER_EXAMPLE_1 = "The NIST Cybersecurity Framework 2.0 emphasizes the integration of cybersecurity into the broader scope of enterprise risk management (ERM). It advocates for a holistic approach, aligning cybersecurity efforts with the organization's overall risk management strategies. The Framework suggests leveraging cybersecurity risk management activities in conjunction with other risk domains, such as financial, legal, operational, and reputational risks. This integration helps organizations balance multiple risk considerations, ensuring a comprehensive view of enterprise risks is maintained."


### STALE PROMPTS ###
CONTEXT_SYSTEM_PROMPT = "you are a chatbot that given a piece of text, gives some context to the human regardin the name and a brief description of the content of a document"

CONTEXT_HUMAN_PROMPT = (
    "give me the name and a very brief description of the document: {text}"
)


REWRITE_INSTRUCTION = """
I want you act as a Prompt Rewriter.\n
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n
But the rewritten prompt must be reasonable and must be understood and responded by humans.\n
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n
You SHOULD complicate the given prompt using the following method: \n{}\n
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n
#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n
""".lstrip()


SUMMARY_SYSTEM_PROMPT = """As a professional summarizer, create a concise and comprehensive summary of the provided text, be it an article, post, conversation, or passage, while adhering to these guidelines:
* Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.
* Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
* Rely strictly on the provided text, without including external information.
* Format the summary in paragraph form for easy understanding.
* do not refer directly to the document, just summarize it
* The language of the summary must be the one defined by the user 
"""
SUMMARY_HUMAN_PROMPT = """
{content} \n\n summarize this text in {language}:
"""
## PROMTP USATI DA DISTILABEL
MUTATION_TEMPLATES = {
    "CONSTRAINTS": REWRITE_INSTRUCTION.format(
        "Please add one more constraints/requirements into '#The Given Prompt#'"
    ),
    "DEEPENING": REWRITE_INSTRUCTION.format(
        "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."
    ),
    "CONCRETIZING": REWRITE_INSTRUCTION.format(
        "Please replace general concepts with more specific concepts."
    ),
    "INCREASED_REASONING_STEPS": REWRITE_INSTRUCTION.format(
        "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."
    ),
    # "BREADTH": CREATE_INSTRUCTION,
}
