from pytest import fixture

from intelligence_layer.core.model import LuminousControlModel
from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.qa.long_context_qa import (
    LongContextQa,
    LongContextQaInput,
)

LONG_TEXT = """Robert Moses''' (December 18, 1888 – July 29, 1981) was an American [[urban planner]] and public official who worked in the [[New York metropolitan area]] during the early to mid 20th century. Despite never being elected to any office, Moses is regarded as one of the most powerful and influential individuals in the history of New York City and New York State. The grand scale of his infrastructural projects and his philosophy of urban development influenced a generation of engineers, architects, and urban planners across the United States.<ref name=":0" />

Moses held various positions throughout his more than forty-year long career. He at times held up to 12 titles simultaneously, including [[New York City Parks Commissioner]] and chairman of the [[Long Island State Park Commission]].<ref>{{Cite web|url=https://www.pbs.org/wnet/need-to-know/environment/the-legacy-of-robert-moses/16018/|title=The legacy of Robert Moses|last=Sarachan|first=Sydney|date=January 17, 2013|website=Need to Know {{!}} PBS|language=en-US|access-date=December 3, 2019}}</ref> Having worked closely with New York governor [[Al Smith]] early in his career, Moses became expert in writing laws and navigating and manipulating the inner workings of state government. He created and led numerous semi-autonomous [[Public authority|public authorities]], through which he controlled millions of dollars in revenue and directly issued [[Bond (finance)|bonds]] to fund new ventures with little outside input or oversight.

Moses's projects transformed the New York area and revolutionized the way cities in the U.S. were designed and built. As Long Island State Park Commissioner, Moses oversaw the construction of [[Jones Beach State Park]], the most visited public beach in the United States,<ref name="Jones Beach">{{cite news |url=http://www.longislandexchange.com/jones-beach.html |website=Long Island Exchange |title=Jones Beach |access-date=November 21, 2012 |archive-url=https://web.archive.org/web/20130121130008/http://www.longislandexchange.com/jones-beach.html |archive-date=January 21, 2013 |url-status=dead }}</ref> and was the primary architect of the [[Parkways in New York|New York State Parkway System]]. As head of the [[MTA Bridges and Tunnels|Triborough Bridge Authority]], Moses had near-complete control over bridges and tunnels in New York City as well as the tolls collected from them, and built, among others, the [[Robert F. Kennedy Bridge|Triborough Bridge]], the [[Brooklyn–Battery Tunnel]], and the [[Throgs Neck Bridge]], as well as several major highways. These roadways and bridges, alongside [[urban renewal]] efforts that saw the destruction of huge swaths of tenement housing and their replacement with large [[New York City Housing Authority|public housing projects]], transformed the physical fabric of New York and inspired other cities to undertake similar development endeavors.

Moses's reputation declined following the publication of [[Robert Caro]]'s [[Pulitzer Prize]]-winning biography ''[[The Power Broker]]'' (1974), which cast doubt on the purported benefits of many of Moses's projects and further cast Moses as racist. In large part because of ''The Power Broker'', Moses is today considered a controversial figure in the history of New York City.

==Early life and career==
Moses was born in [[New Haven, Connecticut]], on December 18, 1888, to [[German Jewish]] parents, Bella (Silverman) and Emanuel Moses.<ref>{{cite news | url=https://www.nytimes.com/learning/general/onthisday/bday/1218.html | title=Robert Moses, Master Builder, is Dead at 92| newspaper=The New York Times |archive-url=https://web.archive.org/web/20160305003155/https://www.nytimes.com/learning/general/onthisday/bday/1218.html |archive-date=March 5, 2016 |url-status=dead}}</ref>{{sfn|Caro|1974|p=25}} He spent the first nine years of his life living at 83 Dwight Street in New Haven, two blocks from [[Yale University]]. In 1897, the Moses family moved to New York City,{{sfn|Caro|1974|pp=29}} where they lived on East 46th Street off Fifth Avenue.<ref>{{cite web |url=http://www.newsday.com/community/guide/lihistory/ny-history-hs722a,0,7092161.story |title=The Master Builder |access-date=April 4, 2007 |last=DeWan |first=George |year=2007 |website=Long Island History |publisher=Newsday |archive-url=https://web.archive.org/web/20061211045554/http://www.newsday.com/community/guide/lihistory/ny-history-hs722a%2C0%2C7092161.story |archive-date=December 11, 2006 |url-status=dead  }}</ref> Moses's father was a successful department store owner and [[real estate]] speculator in New Haven. In order for the family to move to New York City, he sold his real estate holdings and store, then retired.{{sfn|Caro|1974|pp=29}} Moses's mother was active in the [[settlement movement]], with her own love of building. Robert Moses and his brother Paul attended several schools for their elementary and [[secondary education]], including the [[Ethical Culture Fieldston School|Ethical Culture School]], the [[Dwight School]] and the [[Mohegan Lake, New York#Historic places|Mohegan Lake School]], a military academy near [[Peekskill, New York|Peekskill]].{{sfn|Caro|1974|pp=35}}

After graduating from [[Yale College]] (B.A., 1909) and [[Wadham College]], [[Oxford University|Oxford]] (B.A., Jurisprudence, 1911; M.A., 1913), and earning a Ph.D. in [[political science]] from [[Columbia University]] in 1914, Moses became attracted to New York City reform politics.<ref>{{Cite web|url=http://c250.columbia.edu/c250_celebrates/remarkable_columbians/robert_moses.html|title = Robert Moses}}</ref> A committed [[idealism|idealist]], he developed several plans to rid New York of [[Patronage#Politics|patronage hiring]] practices, including being the lead author of a 1919 proposal to reorganize the New York state government. None went very far, but Moses, due to his intelligence, caught the notice of [[Belle Moskowitz]], a friend and trusted advisor to Governor [[Al Smith]].{{sfn|Caro|1974}}  When the state [[Secretary of State of New York|Secretary of State's]] position became appointive rather than elective, Smith named Moses. He served from 1927 to 1929.<ref>{{cite news |date=December 19, 1928 |title=Moses Resigns State Position |url=http://cdsun.library.cornell.edu/cgi-bin/cornell?a=d&d=CDS19281219.2.63.7# |newspaper=Cornell Daily Sun |location=Ithaca, NY |page=8}}</ref>

Moses rose to power with Smith, who was elected as governor in 1918, and then again in 1922. With Smith's support, Moses set in motion a sweeping consolidation of the New York State government. During that period Moses began his first foray into large-scale public work initiatives, while drawing on Smith's political power to enact legislation. This helped create the new [[Long Island State Park Commission]] and the State Council of Parks.<ref>{{cite web|last=Gutfreund|first=Owen|title=Moses, Robert|url=http://www.anb.org/articles/07/07-00375.html|publisher=Anb.org|access-date=December 24, 2014}}</ref> In 1924, Governor Smith appointed Moses chairman of the State Council of Parks and president of the Long Island State Park Commission.<ref>{{Cite book|title=Encyclopedia of the City|url=https://archive.org/details/encyclopediacity00cave|url-access=limited|last=Caves|first=R. W.|publisher=Routledge|year=2004|isbn=978-0-415-25225-6|pages=[https://archive.org/details/encyclopediacity00cave/page/n512 472]}}</ref> This centralization allowed Smith to run a government later used as a model for Franklin D. Roosevelt's [[New Deal]] federal government.{{or|date=October 2022}} Moses also received numerous commissions that he carried out efficiently, such as the development of [[Jones Beach State Park]].{{cn|date=October 2022}} Displaying a strong command of [[law]] as well as matters of [[engineering]], Moses became known for his skill in drafting legislation, and was called "the best bill drafter in [[Albany, New York|Albany]]".<ref name=":0">{{cite news |title=Annals of Power |first=Robert A. |last=Caro |author-link=Robert Caro |url=http://archives.newyorker.com/?i=1974-07-22#folio=032 |magazine=[[The New Yorker]] |date=July 22, 1974 |access-date=September 1, 2011}}</ref> At a time when the public was accustomed to [[Tammany Hall]] corruption and incompetence, Moses was seen as a savior of government.{{sfn|Caro|1974}}

Shortly after [[President of the United States|President]] [[Franklin Delano Roosevelt|Franklin D. Roosevelt's]] [[First inauguration of Franklin D. Roosevelt|inauguration]] in 1933, the [[United States federal government|federal government]] found itself with millions of [[New Deal]] dollars to spend, yet states and cities had few projects ready. Moses was one of the few local officials who had projects [[shovel ready]]. For that reason, New York City was able to obtain significant [[Works Progress Administration]] (WPA), [[Civilian Conservation Corps]] (CCC), and other Depression-era funding. One of his most influential and longest-lasting positions was that of Parks Commissioner of New York City, a role he served from January 18, 1934, to May 23, 1960.<ref>{{Cite web|url=https://www.nycgovparks.org/about/history/commissioners|title=New York City Parks Commissioners : NYC Parks|website=www.nycgovparks.org|language=en|access-date=March 29, 2018}}</ref>

==Offices held==
The many offices and professional titles that Moses held gave him unusually broad power to shape urban development in the New York metropolitan region. These include, according to the New York Preservation Archive Project:<ref>{{Cite web|url=http://www.nypap.org/preservation-history/robert-moses/|title=Robert Moses {{!}}|website=www.nypap.org|language=en-US|access-date=March 29, 2018}}</ref>
*[[Long Island State Park Commission]] (President, 1924–1963)
* New York State Council of Parks (Chairman, 1924–1963)
*[[Secretary of State of New York|New York Secretary of State]] (1927–1928)
* Bethpage State Park Authority (President, 1933–1963)
* Emergency Public Works Commission (Chairman, 1933–1934)
* Jones Beach Parkway Authority (President, 1933–1963)
*[[New York City Department of Parks and Recreation|New York City Department of Parks]] (Commissioner, 1934–1960)
* [[Triborough Bridge]] and Tunnel Authority (Chairman, 1934–1968)
* New York City Planning Commission (Commissioner, 1942–1960)
* New York State Power Authority (Chairman, 1954–1962)
* [[1964 New York World's Fair|New York's World Fair]] (President, 1960–1966)
* Office of the Governor of New York (Special Advisor on Housing, 1974–1975)

==Influence==
During the 1920s, Moses sparred with [[Franklin D. Roosevelt]], then head of the Taconic State Park Commission, who favored the prompt construction of a [[parkway]] through the [[Hudson Valley]]. Moses succeeded in diverting funds to his Long Island parkway projects (the [[Northern State Parkway]], the [[Southern State Parkway]] and the [[Wantagh State Parkway]]), although the [[Taconic State Parkway]] was later completed as well.<ref>{{cite web|url=http://www.nycroads.com/roads/taconic/ |title=Taconic State Parkway |website=NYCRoads.com |access-date=May 25, 2006}}</ref> Moses helped build Long Island's [[Meadowbrook State Parkway]]. It was the first fully divided limited access highway in the world.<ref name="Leonard 1991 339">{{cite book|last=Leonard|first=Wallock|title=The Myth of The Master Builder|year=1991|publisher=Journal of Urban History|page=339}}</ref>

Moses was a highly influential figure in the initiation of many of the reforms that restructured New York state's government during the 1920s. A 'Reconstruction Commission' headed by Moses produced a highly influential report that provided recommendations that would largely be adopted, including the consolidation of 187 existing agencies under 18 departments, a new executive budget system, and the four-year term limit for the governorship.{{sfn|Caro|1974|pp=106, 260}}"""


@fixture
def long_context_qa(luminous_control_model: LuminousControlModel) -> LongContextQa:
    return LongContextQa(model=luminous_control_model)


def test_qa_with_answer(long_context_qa: LongContextQa) -> None:
    question = "What is the name of the book about Robert Moses?"
    input = LongContextQaInput(text=LONG_TEXT, question=question)
    output = long_context_qa.run(input, NoOpTracer())
    assert output.answer
    assert "The Power Broker" in output.answer
    # highlights TODO


def test_qa_with_no_answer(long_context_qa: LongContextQa) -> None:
    question = "Who is the President of the united states?"
    input = LongContextQaInput(text=LONG_TEXT, question=question)
    output = long_context_qa.run(input, NoOpTracer())

    assert output.answer is None


def test_multiple_qa_on_single_task_instance(long_context_qa: LongContextQa) -> None:
    question = "Where was Robert Moses born?"
    input = LongContextQaInput(text=LONG_TEXT, question=question)
    output = long_context_qa.run(input, NoOpTracer())

    input = LongContextQaInput(
        text="This is some arbitrary text without content,", question=question
    )
    output = long_context_qa.run(input, NoOpTracer())

    assert output.answer is None
