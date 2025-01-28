# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:43:35 2024

@author: becker_nic

## CLASSIFICATION MODULE
"""

import os
import pandas as pd
import numpy as np

from tqdm import tqdm
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from general_utils import setup_logger

# Set logging
logger = setup_logger(__name__, 'logfile.log')

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "climate_policy_classification"

class TargetParameterObject(BaseModel):
    # type
    target_parameter: str = Field(description="Classification result after previous consideration of the step-by-step instructions.", enum=["T_Adaptation_C", "T_Adaptation_Unc", "T_Economy_C", "T_Economy_Unc", "T_Netzero", "T_Energy", "T_Transport_C", "T_Transport_O_C", "T_Transport_O_Unc", "T_Transport_Unc", "Not_a_target"])
   

def zero_shot_target_classification_chain(llm):
    """
    Defines the chain for quote classification, including prompt and language model used.
    Designed with single prompt for entire classification logic.

    Parameters
    ----------
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.

    Returns
    -------
    quote_classification_chain : langchain_core.runnables.base.RunnableSequence
        Chain object providing the classification results.

    """
    
    # Prompt
    zero_shot_prompt = PromptTemplate(
                                    template="You are a climate and transport policy analyst, specialized in Nationally Determined Contributions (NDCs). Your task is to classify text from such climate policy documents. You will be provided with quotes from NDCs.\
                                            Use the following step-by-step instructions to classify the quotes:\
                                            Step 1 – Does the quote define a target? If yes, classify as “T”, if not, STOP the process and answer with an explanation, why the quote does not describe any target.\
                                            Step 2 – Does the quote define a net-zero ghg emission target? If yes, classify as “T_Netzero” and STOP the process.\
                                            Step 3 – Which sector is the target concerning about? If it concerns the Energy sector, classify as “T_Energy” and STOP the process. If it concerns the Transport sector, classify as “T_Transport”, if it concerns an economy-wide target, classify as “T_Economy”. If it concerns any other sector, STOP the process and answer naming the sector.\
                                            Step 4 – If the classified sector is Transport, detect if the target is greenhouse gas (GHG) related or not. If it is not GHG related, classify as “T_Transport_O”. If it is GHG related, keep the classification as “T_Transport”. \
                                            Step 5 – If the classified sector is Transport, detect if the target is climate change mitigation related or climate change adaptation related. If it is related to adaptation, classify as “T_Adaptation”, dropping previous results. If it is related to mitigation, keep the classification as “T_Transport”.\
                                            Step 6 – Does the quote contain any conditionality related to the defined target? If yes, end the classification with a suffix “_C”. If there is no conditionality defined for the target, add the suffix “_Unc”.\
                                            Answer by ONLY returning the suggested complete classification.\
                                            Possible Answers: T_Adaptation_C, T_Adaptation_Unc, T_Economy_C, T_Economy_Unc, T_Energy, T_Transport_C, T_Transport_O_C, T_Transport_O_Unc, or T_Transport_Unc.\
                                            Before answering, check if the provided answer corresponds to one of these terms.\
                                            \nHere is the quote: {quote}",
                                    input_variables=["quote"],
    )
    quote_classification_chain = zero_shot_prompt | llm.with_structured_output(TargetParameterObject)
    
    return quote_classification_chain


@traceable(
    task="classification",
    version="zero-shot single prompt"
    )
def classify_target_quotes(quotes_dict, chain):
    """
    Assigns classification labels to the extracted quotes.
    Possible labels are:
        T_Adaptation_C, T_Adaptation_Unc, T_Economy_C, T_Economy_Unc, T_Energy, T_Transport_C, T_Transport_O_C, T_Transport_O_Unc, or T_Transport_Unc.
    In case none of the labels apply, the model provides an explanation.

    Parameters
    ----------
    quotes_dict : dict
        Dictionary containing quotes for each retrieved document, along with respective metadata.
    chain : langchain_core.runnables.base.RunnableSequence
        Chain object containing prompt and language model for classification.

    Returns
    -------
    output_df : pandas DataFrame
        Dataframe containing an entry for each quote, with the respective class label assignation.

    """
    # initialize output dataframe
    output_df = pd.DataFrame(columns=['document', 'page', 'info_type', 'keywords', 'quote', 'class'])
    for key, value in tqdm(quotes_dict.items()):
        
        if type(value) == str:
            continue

        # check if quotes available
        if len(value['quotes'])>0:
            
            for q in value['quotes']:
                #create entry for each quote
                entry = []
                entry.append(value['filename'])
                entry.append(int(value['page_number']))
                entry.append(value['type'])
                entry.append(str(value['keywords']))
                entry.append(q[1])
                entry.append('NONE') #placeholder for class         
                output_df.loc[len(output_df.index)] = entry

    X = output_df.quote
    y = [chain.invoke({"quote": x}).target_parameter for x in X]
    output_df['class'] = pd.Series(y)

    return output_df

class QuoteTypeObject(BaseModel):
    # type
    target: str = Field(description="In an NDC, a 'target' is defined as a quantifiable declaration of intend, with a proposed target year. Does the text refer to a 'target'?", enum=["True", "False"])
    measure: str = Field(description="In an NDC, a 'measure' is defined as an action that is planned to be undertaken, without necessarily stating a target year for completion. Does the text refer to a measure in the transport sector?", enum=["True", "False"])

class TargetObject(BaseModel):
    # sector
    energy: str = Field(description="Does the text refer to the ENERGY sector?", enum=["True", "False"])
    transport: str = Field(description="Does the text refer to the TRANSPORT sector or a TRANSPORT-RELATED subsector?", enum=["True", "False"])
    #economy_wide: str = Field(description="A reduction target for greenhouse gas emissions (covering CO2 and other relevant greenhouse gases) has been set for the whole economy or collectively for all sectors covered. Does not deal with a specific sector, but with the economy in general.", enum=["True", "False"]) # not performing well, after first evaluation and comparison to single prompt
    economy_wide: str = Field(description="Does the text not refer to a specific sector, but to the ECONOMY in general?", enum=["True", "False"]) # revisited after comparison with single-prompt results --> not yet evaluated
    # mitigation / adaptation
    mitigation: str = Field(description="Does the text concern climate change MITIGATION in a direct or indirect manner?", enum=["True", "False"])
    adaptation: str = Field(description="Does the text concern climate change ADAPTATION or building resilience against consequencs of climate change?", enum=["True", "False"])
    # GHG
    ghg: str = Field(description="Does the text concern GREENHOUSE GAS EMISSIONS in a direct manner?", enum=["True", "False"])
    # Net-Zero
    net_zero: str = Field(description="Does the text define a NET-ZERO ghg emission target?", enum=["True", "False"])
    # conditionality
    conditional: str = Field(description="Does the text define a CONDITIONALITY or contingency upon other developments?", enum=["True", "False"])
    unconditional: str = Field(description="This text does NOT define a CONDITIONALITY or contingency upon any other developments.", enum=["True", "False"])

class MeasureTypeObject(BaseModel):
    # type
    mitigation_measure: str = Field(description="In an NDC, a 'mitigation' measure is defined as an action that is planned to be undertaken to mitigate the effects of cliamte change, such as transport system improvements, low-carbon fuels, mode shift and demand management. Does the text refer to a mitigation measure in the transport sector?", enum=["True", "False"])
    adaptation_measure: str = Field(description="In an NDC, an 'adaptation' measure is defined as an action that is planned to be undertaken to adapt to the effects of climate change, such as the investment in resilient infrastructure, information management and regulatory planning. Does the text refer to an adaptation measure in the transport sector?", enum=["True", "False"])

class MitigationObject_TSI(BaseModel):
    # MITIGATION MEASURES
    # high-level category: Transport System Improvements 
    A_Complan: str = Field(description="Measure concerning General transport planning", enum=["True", "False"])
    A_Natmobplan: str = Field(description="Measure concerning National mobility plans", enum=["True", "False"])
    A_SUMP: str = Field(description="Measure concerning Sustainable urban mobility plans", enum=["True", "False"])
    A_LATM: str = Field(description="Measure concerning Traffic management", enum=["True", "False"])
    A_Landuse: str = Field(description="Measure concerning General land use", enum=["True", "False"])
    A_Density: str = Field(description="Measure concerning Land use Development density or intensiveness", enum=["True", "False"])
    A_Mixuse: str = Field(description="Measure concerning Mixed land use", enum=["True", "False"])
    S_Infraimprove: str = Field(description="Measure concerning General infrastructure improvements", enum=["True", "False"])
    S_Infraexpansion: str = Field(description="Measure concerning Expansion of infrastructure", enum=["True", "False"])
    S_Intermodality: str = Field(description="Measure concerning Intermodality", enum=["True", "False"])
    I_Freighteff: str = Field(description="Measure concerning General freight efficiency improvements", enum=["True", "False"])
    I_Load: str = Field(description="Measure concerning Improving freight load efficiency", enum=["True", "False"])
    S_Railfreight: str = Field(description="Measure concerning Freight transport shifting to rail or inland waterways", enum=["True", "False"])
    I_Education: str = Field(description="Measure concerning General education and behavior change", enum=["True", "False"])
    I_Ecodriving: str = Field(description="Measure concerning Ecodriving", enum=["True", "False"])
    I_Capacity: str = Field(description="Measure concerning Sustainable transport capacity building", enum=["True", "False"])
    I_Campaigns: str = Field(description="Measure concerning Ecucational Campaigns", enum=["True", "False"])

class MitigationObject_MSDM(BaseModel):
    # MITIGATION MEASURES
    # high-level category: Mode shift and demand management
    A_TDM: str = Field(description="Measure concerning General transport demand management", enum=["True", "False"])
    S_Parking: str = Field(description="Measure concerning General parking measures", enum=["True", "False"])
    A_Parkingprice: str = Field(description="Measure concerning Parking Pricing", enum=["True", "False"])
    A_Caraccess: str = Field(description="Measure concerning Car access restriction zones", enum=["True", "False"])
    A_Commute: str = Field(description="Measure concerning Commuter trip reduction policies", enum=["True", "False"])
    A_Work: str = Field(description="Measure concerning Alternative work schedules (flextime, staggered shifts, compressed work week)", enum=["True", "False"])
    A_Teleworking: str = Field(description="Measure concerning Teleworking", enum=["True", "False"])
    A_Economic: str = Field(description="Measure concerning General economic instruments", enum=["True", "False"])
    A_Emistrad: str = Field(description="Measure concerning Emissions trading and carbon pricing", enum=["True", "False"])
    A_Finance: str = Field(description="Measure concerning Financial instruments to support decarbonisation", enum=["True", "False"])
    A_Procurement: str = Field(description="Measure concerning Green public procurement", enum=["True", "False"])
    A_Fossilfuelsubs: str = Field(description="Measure concerning Fossil fuel subsidy elimination", enum=["True", "False"])
    A_Fueltax: str = Field(description="Measure concerning Fuel tax", enum=["True", "False"])
    A_Vehicletax: str = Field(description="Measure concerning Vehicle taxes", enum=["True", "False"])
    A_Roadcharging: str = Field(description="Measure concerning Road charging and tolls", enum=["True", "False"])
    S_PublicTransport: str = Field(description="Measure concerning General public transport improvement", enum=["True", "False"])
    S_PTIntegration: str = Field(description="Measure concerning Public transit integration and expansion", enum=["True", "False"])
    S_PTPriority: str = Field(description="Measure concerning Express lanes/ public transport priority", enum=["True", "False"])
    S_BRT: str = Field(description="Measure concerning BRT", enum=["True", "False"])
    S_Activemobility: str = Field(description="Measure concerning General active mobility", enum=["True", "False"])
    S_Walking: str = Field(description="Measure concerning Walking", enum=["True", "False"])
    S_Cycling: str = Field(description="Measure concerning Cycling", enum=["True", "False"])
    S_Sharedmob: str = Field(description="Measure concerning General shared mobility", enum=["True", "False"])
    S_Ondemand: str = Field(description="Measure concerning On-demand transport", enum=["True", "False"])
    S_Maas: str = Field(description="Measure concerning Mobility-as-a-Service (MaaS)", enum=["True", "False"])
    I_Other: str = Field(description="Measure concerning General innovations and digitalization in transport", enum=["True", "False"])
    I_ITS: str = Field(description="Measure concerning Intelligent transport systems", enum=["True", "False"])
    I_Autonomous: str = Field(description="Measure concerning Autonomous vehicles (AVs)", enum=["True", "False"])
    I_DataModelling: str = Field(description="Measure concerning Data & modelling improvements", enum=["True", "False"])

class MitigationObject_LCF(BaseModel):
    # MITIGATION MEASURES
    # high-level category: Low-carbon fuels and energy vectors
    I_Vehicleimprove: str = Field(description="Measure concerning General vehicle improvements", enum=["True", "False"])
    I_Fuelqualimprove: str = Field(description="Measure concerning Fuel quality improvements", enum=["True", "False"])
    I_Inspection: str = Field(description="Measure concerning Inspection and maintenance", enum=["True", "False"])
    I_Efficiencystd: str = Field(description="Measure concerning Vehicle air pollution emission standards", enum=["True", "False"])
    I_Vehicleeff: str = Field(description="Measure concerning Vehicle efficiency standards", enum=["True", "False"])
    A_LEZ: str = Field(description="Measure concerning Low emission zones", enum=["True", "False"])
    I_VehicleRestrictions: str = Field(description="Measure concerning Vehicle restrictions (import, age, access, sale, taxation)", enum=["True", "False"])
    I_Vehiclescrappage: str = Field(description="Measure concerning Vehicle scrappage scheme", enum=["True", "False"])
    I_Lowemissionincentive: str = Field(description="Measure concerning Low emission vehicle purchase incentives", enum=["True", "False"])
    I_Altfuels: str = Field(description="Measure concerning General alternative fuels", enum=["True", "False"])
    I_Ethanol: str = Field(description="Measure concerning Ethanol", enum=["True", "False"])
    I_Biofuel: str = Field(description="Measure concerning Biofuels", enum=["True", "False"])
    I_LPGCNGLNG: str = Field(description="Measure concerning LPG/CNG/LNG", enum=["True", "False"])
    I_Hydrogen: str = Field(description="Measure concerning Hydrogen", enum=["True", "False"])
    I_RE: str = Field(description="Measure concerning Use of renewable energy", enum=["True", "False"])
    I_Transportlabel: str = Field(description="Measure concerning General transport labels", enum=["True", "False"])
    I_Efficiencylabel: str = Field(description="Measure concerning Efficiency labels", enum=["True", "False"])
    I_Freightlabel: str = Field(description="Measure concerning Green freight labels", enum=["True", "False"])
    I_Vehiclelabel: str = Field(description="Measure concerning Vehicle labelling", enum=["True", "False"])
    I_Fuellabel: str = Field(description="Measure concerning Fuel labelling", enum=["True", "False"])

class MitigationObject_EI(BaseModel):
    # MITIGATION MEASURES
    # high-level category: Electrification 
    I_Emobility: str = Field(description="Measure concerning General e-mobility", enum=["True", "False"])
    I_Emobilitycharging: str = Field(description="Measure concerning Charging Infrastructure for Electric Vehicles (EVs)", enum=["True", "False"])
    I_Smartcharging: str = Field(description="Measure concerning Smart charging policies", enum=["True", "False"])
    I_Emobilitypurchase: str = Field(description="Measure concerning Purchase incentives for Electric Vehicles (EVs)", enum=["True", "False"])
    I_ICEdiesel: str = Field(description="Measure concerning ICE (gasoline and diesel) bans", enum=["True", "False"])
    S_Micromobility: str = Field(description="Measure concerning General micromobility", enum=["True", "False"])
    # high-level category: Innovation and up-scaling 
    I_Aviation: str = Field(description="Measure concerning General aviation improvements", enum=["True", "False"])
    I_Aircraftfleet: str = Field(description="Measure concerning Aircraft fleet renovation", enum=["True", "False"])
    I_CO2certificate: str = Field(description="Measure concerning Airport CO2 certification", enum=["True", "False"])
    I_Capacityairport: str = Field(description="Measure concerning Environmental capacity constraints on airports", enum=["True", "False"])
    I_Jetfuel: str = Field(description="Measure concerning Jet fuel policies", enum=["True", "False"])
    I_Airtraffic: str = Field(description="Measure concerning Air traffic management", enum=["True", "False"])
    I_Shipping: str = Field(description="Measure concerning General shipping improvement", enum=["True", "False"])
    I_Onshorepower: str = Field(description="Measure concerning Support of on-shore power and electric charging facilities in ports", enum=["True", "False"])
    I_PortInfra: str = Field(description="Measure concerning Port infrastructure improvements", enum=["True", "False"])
    I_Shipefficiency: str = Field(description="Measure concerning Ship efficiency improvements", enum=["True", "False"])
    
class AdaptationObject(BaseModel):
    # ADAPTATION MEASURES
    R_System: str = Field(description="Measure concerning efforts to adapt transport system and infrastructure to climate change impacts and to increase its resilience", enum=["True", "False"])
    R_Maintain: str = Field(description="Measure concerning general efforts to repair or maintain transport infrastructure, without reference to climate change adaptation.", enum=["True", "False"])
    R_Risk: str = Field(description="Measure concerning risk assessments or efforts to understand risks and impacts to the transport system (e.g. through modelling).", enum=["True", "False"])
    R_Tech: str = Field(description="Measure concerning efforts to adopt resilient transport technologies (e.g. climate resilient materials for streets or cars).", enum=["True", "False"])
    R_Monitoring: str = Field(description="Measure concerning efforts to adopt monitoring systems, e.g. to detect risks early on.", enum=["True", "False"])
    R_Inform: str = Field(description="Measure concerning efforts to adopt notification systems, e.g. to inform drivers about flooding, so they can take alternate routes.", enum=["True", "False"])
    R_Emergency: str = Field(description="Measure concerning emergency and disaster planning that is specifically related to transport.", enum=["True", "False"])
    R_Education: str = Field(description="Measure concerning efforts to educate and train transport officials regarding the vulnerability of transport systems and infrastructure to climate change.", enum=["True", "False"])
    R_Warning: str = Field(description="Measure containing explicit mention of an early warning system.", enum=["True", "False"])
    R_Planning: str = Field(description="Measure concerning activities designed to raise the importance of resilience and adaptation in transport planning.", enum=["True", "False"])
    R_Relocation: str = Field(description="Measure concerning efforts to relocate infrastructure or populations due to current or anticipated threats.", enum=["True", "False"])
    R_Redundancy: str = Field(description="Measure concerning construction of redundant infrastructure/facilities, to prepare for the possible failure of existing systems.", enum=["True", "False"])
    R_Disinvest: str = Field(description="Measure concerning measures to discontinue or avoid expanding transport services or infrastructure. Abandonment or disinvestment of infrastructure", enum=["True", "False"])
    R_Laws: str = Field(description="Measure concerning laws, programmes or regulations that focus on climate change adaptation in the transport sector.", enum=["True", "False"])
    R_Design: str = Field(description="Measure concerning the adoption of improved, more resilient design standards to effectively protect or reinforce transport facilities or infrastructure.", enum=["True", "False"])
    R_Other: str = Field(description="Measure concerning Other adaptation measures for transport not falling under the categories listed above.", enum=["True", "False"])

class MitigationObject_TSI_FewShot(BaseModel):
    # MITIGATION MEASURES
    # high-level category: Transport System Improvements 
    A_Complan: str = Field(description="Measure concerning any general mention of transport planning", enum=["True", "False"])
    A_Natmobplan: str = Field(description="Measure concerning National mobility plans. A national mobility plan provides a comprehensive long-term planning framework for the transport sector. It features a vision and timeframes for action at the national level. This parameter records any activities that focus on nationwide transport.", enum=["True", "False"])
    A_SUMP: str = Field(description="Measure concerning Sustainable urban mobility plans (SUMPs). SUMP is “a strategic plan designed to satisfy the mobility needs of people and businesses in cities and their surroundings for a better quality of life. It builds on existing planning practices and takes due consideration of integration, participation and evaluation principles”.", enum=["True", "False"])
    A_LATM: str = Field(description="Measure concerning Traffic management, looks at management, infrastructure and techonlogical approaches with the goal of improving traffic flow. <example> Improve traffic management, planning & Infrastructure: Improve traffic management, planning together with urban planning </example>", enum=["True", "False"])
    #A_Landuse: str = Field(description="Measure concerning General land use", enum=["True", "False"])
    #A_Density: str = Field(description="Measure concerning Land use Development density or intensiveness. <example> functional mix and spatial density to bring living and working closer together </example>", enum=["True", "False"])
    A_Mixuse: str = Field(description="Measure concerning mixed land use. Captures different features of the urban built environment that contribute to low mileage and sustainble travel patterns. This inclues urban density, land-use diversity 'mixed used' and enhanced accessibility. <example>adoption of integrated land use and transportation systems that will connect housing, jobs, schools, and communities through a variety of integrated low-carbon mobility solutions. </example>", enum=["True", "False"])
    #A_Urban: str = Field(description="Measure concerning urban planning. Captures different features of the urban built environment that contribute to low mileage and sustainble travel patterns. This inclues urban density, land-use diversity 'mixed used' and enhanced accessibility.", enum=["True", "False"]) # NEW: summarizes the above three previous indicators
    S_Infraimprove: str = Field(description="Measure concerning general infrastructure improvements. This category is for measures that outline general improvements in transport infrastructure or the transport system as a whole, without providing details about specific measures. <example> Upgrading of roads to improve connectivity, reduce travel time and vehicle emissions </example>", enum=["True", "False"])
    S_Infraexpansion: str = Field(description="Measure concerning expansion of infrastructure. This category refers to activities that aim to introduce new infrastructure or expand infrastructure for transport. If a measure is dedicated to a specific transport mode, then it might be captured under that specific category and NOT here. Any general mention of expanding transport infrastructure is collected here. <example> Actively develop large-capacity and high-efficiency inter-regional rapid passenger transport services with high-speed rail and aviation as the mainstay, </example>", enum=["True", "False"])
    S_Intermodality: str = Field(description="Measure concerning Intermodality. Intermodality is the combination of different transport modes with the goal of enabling convenient, seamless transfer between them. Any general activities that highlight intermodality but do not specify actions are included here.", enum=["True", "False"])
    I_Freighteff: str = Field(description="Measure concerning General freight efficiency improvements. This category records general efficiency improvements in freight. If the measure does not specify a specific activity or action that belongs to the other freight efficiency activities, then it is included here. <example> To design a plan for the technological efficiency improvement of the cargo transport sector, which will consider aspects such as technological improvement (LPG, for example), the use of filters, biofuels and other efficiency improvements.</example>", enum=["True", "False"])
    I_Load: str = Field(description="Measure concerning improving freight load efficiency. This category is for measures that encourage reliance on high-capacity vehicles (trains, ships, etc.) in order to achieve lower carbon intensity per ton transported.", enum=["True", "False"])
    S_Railfreight: str = Field(description="Measure concerning Freight transport shifting to rail or inland waterways.", enum=["True", "False"])
    I_Education: str = Field(description="Measure concerning General education and behavior change. This category is for general educational activities and behavioral change related to transport, e.g. concerning the environmental impacts of private vehicle use, the benefits of electric vehicles, etc.", enum=["True", "False"])
    I_Ecodriving: str = Field(description="Measure concerning Ecodriving. Ecodriving refers to educational measures that encourage more efficient driving practices. Such practices can reduce fuel consumption and are captured in this category. <example>considering the diverse needs of local transportation for low-speed driving </example>", enum=["True", "False"])
    #I_Capacity: str = Field(description="Measure concerning Sustainable transport capacity building", enum=["True", "False"])
    #I_Campaigns: str = Field(description="Measure concerning Ecucational Campaigns", enum=["True", "False"])

class MitigationObject_MSDM_FewShot(BaseModel):
    # MITIGATION MEASURES
    # high-level category: Mode shift and demand management
    A_TDM: str = Field(description="Measure concerning General transport demand management. Any general mention of activities focusing on reducing demand for motorized transport.", enum=["True", "False"])
    S_Parking: str = Field(description="Measure concerning Parking. Refers to actions that aim to improve parking management such as pricing, quantity restrictions, parking reform etc. which then also contribute to a reduced demand of motorised travel.  <example>Develop park and ride infrastructure developments</example>", enum=["True", "False"])
    #A_Parkingprice: str = Field(description="Measure concerning Parking Pricing", enum=["True", "False"])
    A_Caraccess: str = Field(description="Measure concerning Car access restriction zones, that restrict the physical access of certain types of vehicles to certain places (e.g. city centres). An example are Low Emission Zones. By resticting access based on certain criteria (type of propulsion technique, euro-Standards, age etc) access limitations can have differentiated impact such as GHG emissions reductions, air pollutants reduction and congestion prevention. They can encourage the use of low-carbon modes like cycling and public transport and/or support technological shift to cleaner technologies. ", enum=["True", "False"])
    A_Commute: str = Field(description="Measure concerning commuter trip reduction policies. Management of circumstances and incentives for employee commuter travel and working arrangements to reduce traffic and automobile use. <example> To create conditions favorable to improvement the organization of labor relations at the national and local level in order to reduce the need for employees travel by transport vehicles </example>", enum=["True", "False"])
    #A_Work: str = Field(description="Measure concerning Alternative work schedules (flextime, staggered shifts, compressed work week)", enum=["True", "False"])
    #A_Teleworking: str = Field(description="Measure concerning Teleworking. <example>More widespread use of digital communication allows more flexibility in where people work from, reduces the need for transport and gives people more free time. </example>", enum=["True", "False"])
    A_Economic: str = Field(description="Measure concerning General economic instruments. Concerns any general mention of economic instruments, such as taxes & tax breaks, fees, duties and subsidies, which contribute to integrating environmental costs and benefits into the budgets of households and firms.", enum=["True", "False"])
    A_Emistrad: str = Field(description="Measure concerning Emissions trading and carbon pricing. This category refers to emissions trading systems (ETS) or cap-and-trade systems: a pricing mechanism for emitted greenhouse gas emissions. Unlike a direct carbon tax, where the unit price of CO2 is fixed, under an emissions trading scheme, the price per tonne of CO2 varies. The overall amount of emissions is fixed for a given period of time (e.g. annually). Entities are allocated a set amount of CO2 emissions allowances, or quotas, and trade emissions with each another. Those able to reduce their emissions below their allowance level can trade them with those emitting in excess of their allowance.", enum=["True", "False"])
    A_Finance: str = Field(description="Measure concerning Financial instruments to support decarbonisation. This category records financing instruments used to pay for technologies, projects and programmes that reduce GHG emissions. Financing instruments that aim to support decarbonization include climate finance solutions, investments in EVs, green bonds, etc. Not to confound with economic instruments! <example>The government's existing funding programmes designed to shift transport from the road to the railways or inland waterways will be strengthened.</example>", enum=["True", "False"])
    A_Procurement: str = Field(description="Measure concerning Green public procurement. This category refers incentives to stakeholders to taking into account environmental impacts when procuring goods and services. Applied to transport, it means that a public authority can develop green procurement regulations that, for example, only allow the purchase of zero-emission vehicles. Such measures can support the transition to cleaner public vehicle fleets and more sustainable consumption. <example>Gradually reduce the procurement of public transport vehicles using fossil fuels with high greenhouse gas emissions, from public funds </example>", enum=["True", "False"])
    A_Fossilfuelsubs: str = Field(description="Measure concerning Fossil fuel subsidy elimination. This category refers to policies and decisions that eliminate or reduce subsidies for fossil fuels. Energy subsidies are used by governments to lower the cost of producing or consuming fossil fuels. Eliminating such subsidies can help to reduce reliance on fossil fuels.", enum=["True", "False"])
    A_Fueltax: str = Field(description="Measure concerning Fuel tax. This category records national or local taxes on the sale of fuel. Every fuel type is taxed differently. One target of taxing fuel is to reduce fuel consumption and encourage more efficient transport modes.", enum=["True", "False"])
    A_Vehicletax: str = Field(description="Measure concerning Vehicle taxes. This refers to taxes on vehicle pruchase and/or ownership. <example> incentives for purchasing more efficient vehicles through (...) import tariffs </example>", enum=["True", "False"])
    A_Roadcharging: str = Field(description="Measure concerning Road charging and tolls. Refers to surcharges applied to general or specific road use, including in particular highway tolls. This includes congestion pricing.", enum=["True", "False"])
    S_PublicTransport: str = Field(description="Measure concerning General Public Transport improvement. This covers all activities that aim to improve the public transport system. <example>We are working with industry to modernise fares ticketing and retail and encourage a shift to rail and cleaner and greener transport journeys.</example>", enum=["True", "False"])
    S_PTIntegration: str = Field(description="Measure concerning Public transit integration and expansion. This covers activities that aim to expand public transport or integrate different public transport services into a single system. <example>Increasing the share of public transport, including railways </example>", enum=["True", "False"])
    S_PTPriority: str = Field(description="Measure concerning public transport priority. Refers to actions that give priority to public transport over other modes. Examples include transit signal priorities, access priority, intelligent transport systems and express lanes. <example> The UAE will also provide priority lanes for buses and toll adjustments to benefit group transports and to incentivise the uptake of public and shared transportation.</example>", enum=["True", "False"])
    S_BRT: str = Field(description="Measure concerning Bus Rapid Transit (BRT). This refers to a bus system with high speed, capacity, punctuality and operating flexibility. Common characteristics of a BRT system include the use of bus-only lanes, advanced ticketing, and articulated buses.", enum=["True", "False"])
    S_Activemobility: str = Field(description="Measure concerning general Active Mobility. This category is used for general measures that refer to walking and cycling. <example>active mobility (bicycle and foot traffic) promoted by corresponding infrastructure and made accessible by attractive mobility services </example>", enum=["True", "False"])
    S_Walking: str = Field(description="Measure concerning Walking. This category covers any action that specifically mentions improving walking. <example>Moreover, the encouragement of use of bicycles and construction of designated lanes and other infrastructure </example>", enum=["True", "False"])
    S_Cycling: str = Field(description="Measure concerning Cycling. This category covers any action that specifically mentions improving cycling. <example> accessible pedestrian routes </example>", enum=["True", "False"])
    S_Sharedmob: str = Field(description="Measure concerning General shared mobility. This includes general measures in the area of shared mobility, such as bike sharing, car-sharing, shared scooters, etc.", enum=["True", "False"])
    #S_Ondemand: str = Field(description="Measure concerning On-demand transport", enum=["True", "False"])
    #S_Maas: str = Field(description="Measure concerning Mobility-as-a-Service (MaaS)", enum=["True", "False"])
    I_Other: str = Field(description="Measure concerning General innovations and digitalization in transport. This category includes activities that mention the use of innovation and digitalization to improve the efficiency of transport. <example>promote research and innovation and help urban transport flow more freely and cleanly </example>", enum=["True", "False"])
    I_ITS: str = Field(description="Measure concerning Intelligent Transport Systems (ITS). This refers to transport systems that harness technology to improve the management and operation of transport services. Relevant technologies include sensors, wireless communications, notification systems and other ICT solutions.", enum=["True", "False"])
    I_Autonomous: str = Field(description="Measure concerning Autonomous Vehicles (AVs). This refers to measures that promote self-driving vehicles, artificial intelligence and any other mechanisms that support the automation of passenger and freight transport.", enum=["True", "False"])
    I_DataModelling: str = Field(description="Measure concerning Data & modelling improvements. Refers to measures related to transport data (e.g. collection, analysis or application) as well as models designed to predict traffic flows or transport demand growth.", enum=["True", "False"])

class MitigationObject_LCF_FewShot(BaseModel):
    # MITIGATION MEASURES
    # high-level category: Energy efficiency
    I_Vehicleimprove: str = Field(description="Measure concerning General vehicle improvements", enum=["True", "False"])
    I_Fuelqualimprove: str = Field(description="Measure concerning Fuel quality improvements. A high-quality fuel contains very low levels of sulfur. Countries set fuel quality standards in order to guarantee fuel quality. This category covers any mention of clean fuels or better fuel quality in the transport sector. <example> Improve the quality of gasoline and new types of alternative fuels </example>", enum=["True", "False"])
    I_Inspection: str = Field(description="Measure concerning Inspection and maintenance of vehicles. A well-maintained vehicle can ensure higher energy efficiency. <example> Conducting statutory tests on 30% of on-road vehicles by 2030, and 60% by 2040. </example>", enum=["True", "False"])
    I_Efficiencystd: str = Field(description="Measure concerning Vehicle air pollution emission standards. This captures emission standards that regulate air pollution exhaust emission (such as NOx) such as the EURO standards Euro1-6. Not referring to CO2 standards!", enum=["True", "False"])
    I_Vehicleeff: str = Field(description="Measure concerning Vehicle efficiency standards. This refers to measures designed to improve vehicle efficiency or lower transport emissions. This is done through fuel economy (or CO2) standards, which regulate how far a vehicle must travel when consuming a given quantity of fuel (e.g. in liters per 100 km or miles per gallon or CO2/100km).", enum=["True", "False"])
    #A_LEZ: str = Field(description="Measure concerning Low emission zones. <example>Introduce low-emission zones in municipalities, including charging for entry into these zones and traffic calming in settlements (the introduction of functional 30 zones and cycling streets, including transport-technical facilities).</example>", enum=["True", "False"])
    I_VehicleRestrictions: str = Field(description="Measure concerning Vehicle restrictions (import, age, access, sale, taxation). This encompasses various restrictions to vehicle ownership or purchase, including import bans on older vehicles or sale restrictions on particularly polluting vehicles.", enum=["True", "False"])
    I_Vehiclescrappage: str = Field(description="Measure concerning Vehicle scrappage scheme. In order to support the transition to cleaner, more efficient vehicles, governments may provide incentives when an owner scraps their current, old vehicle (rather than reselling it). <example> Support vehicle renewal to accelerate the energy transition, while taking the economic impacts of this into account and paying particular attention to the most precarious and geographically isolated members of the population </example>", enum=["True", "False"])
    #I_Transportlabel: str = Field(description="Measure concerning General transport labels", enum=["True", "False"])
    I_Efficiencylabel: str = Field(description="Measure concerning Efficiency labels. This refers to measures requiring publication of information on environment impacts, this environmental impact can be GHG emissions, fuel consumption, carbon intesntiy of a fuel or local pollutants. All of the different labels are captured under this parameter. ", enum=["True", "False"])
    #I_Freightlabel: str = Field(description="Measure concerning Green freight labels. <example> The Baku port was awarded the “EcoPorts” certificate of the European Sea Ports Organization. </example>", enum=["True", "False"])
    #I_Vehiclelabel: str = Field(description="Measure concerning Vehicle labelling", enum=["True", "False"])
    #I_Fuellabel: str = Field(description="Measure concerning Fuel labelling", enum=["True", "False"])

    # high-level category: Alternative fuels
    I_Lowemissionincentive: str = Field(description="Measure concerning Low emission vehicle purchase incentives. This refers to purchase incentives granted to consumers for lower emission vehicles (__excluding electric and hybrid vehicles!__).", enum=["True", "False"])
    I_Altfuels: str = Field(description="Measure concerning General alternative fuels. Any general reference to the use of alternative fuels in the transport sector is recorded here. <example> Transport emissions can be reduced both through a greener selection of fuels </example>", enum=["True", "False"])
    #I_Ethanol: str = Field(description="Measure concerning Ethanol", enum=["True", "False"])
    I_Biofuel: str = Field(description="Measure concerning Biofuels. Conventional diesel and gasoline can be mixed with less carbon-intense fuels. Many national governments set blending mandates (for example, 10% or 20% of diesel has to be biofuel). The most common biofule is ethanol. Any general biofuel blending mandates as well as specific methons of ethanol (or other biofuls) are covered here.", enum=["True", "False"])
    I_LPGCNGLNG: str = Field(description="Measure concerning liquified petroleum gas (LPG), compressed natural gas (CNG) or liquified natural gas (LNG) in the transport sector.", enum=["True", "False"])
    I_Hydrogen: str = Field(description="Measure concerning Hydrogen. Hydrogen is used in fuel-cell electric vehicles. Green hydrogen that is produced using renewable electricity is seen as one important component of the energy transition in transport.", enum=["True", "False"])
    I_RE: str = Field(description="Measure concerning use of Renewable Energy. Renewable energy for transport looks at the use of biofuels, green hydrogen and green electricity. This refers to actions that make a direct link between transport and renewables.", enum=["True", "False"])
    I_ICEdiesel: str = Field(description="Measure concerning ICE (gasoline and diesel) bans. This entails efforts to phase-out of fossil fuel vehicles. The most common policy is a sales ban on new diesel or gasoline vehicles starting in a specific year. Such policies seek to accelerate the adoption of electric vehicles.", enum=["True", "False"])
    
class MitigationObject_EI_FewShot(BaseModel):
    # MITIGATION MEASURES
    # high-level category: Electrification 
    I_Emobility: str = Field(description="Measure concerning General e-Mobility. Any general policies that refer to electric mobility without specifying a transport mode or specific measure are covered by this category. <example>simplify administrative procedures in transport electrification. </example>", enum=["True", "False"])
    I_Emobilitycharging: str = Field(description="Measure concerning Charging Infrastructure for Electric Vehicles (EVs). Electric vehicle charging infrastructure is needed to promote the adoption of electric vehicles. Measures that seek to increase the number of public charging stations or facilitate more private/public charging points are covered here. <example> A few rapid charging stations can be setup around the island powered by hydro or wind to provide the daily charging requirement, with solar PV reserved for top-up at bus stops and parking garages.</example>", enum=["True", "False"])
    I_Smartcharging: str = Field(description="Measure concerning Smart charging policies. Smart charging refers to systems that optimize electric vehicle charging by prioritizing off-peak hours or times of high variable renewable feed-in.", enum=["True", "False"])
    I_Emobilitypurchase: str = Field(description="Measure concerning Purchase incentives for Electric Vehicles (EVs). National and local governments can support the transition to e-mobility by providing financial incentives for the purchase of electric vehicles.", enum=["True", "False"])
    S_Micromobility: str = Field(description="Measure concerning General Micromobility. Micromobility refers to electric personal transportation devices, such as electric kick-scooters and other electric-powered devices, not covered under shared mobility. <example> …,creation and expansion of infrastructure for..., electric scooters, </example>", enum=["True", "False"])
    # high-level category: Innovation and up-scaling 
    I_Aviation: str = Field(description="Measure concerning general Aviation improvements. Any general measures that focus on the aviation sector are referred to here. <example>Reduce aviation emissions.</example>", enum=["True", "False"])
    I_Aircraftfleet: str = Field(description="Measure concerning Aircraft fleet renovation. Newer aircraft are generally more energy efficient. This category refers to activities designed to renew the aircraft fleet or only allow newer aircraft to operate.", enum=["True", "False"])
    I_CO2certificate: str = Field(description="Measure concerning Airport CO2 certification. CO2 certification systems aim to mitigate greenhouse gas emissions by airports and ground operations. This parameter is for initiatives designed to improve the energy efficiency and carbon footprint of airports. <example> The level 3 accreditation status for Hamad International Airport is another major initiative towards sustainability. </example>", enum=["True", "False"])
    #I_Capacityairport: str = Field(description="Measure concerning Environmental capacity constraints on airports", enum=["True", "False"])
    I_Jetfuel: str = Field(description="Measure concerning Jet fuel policies. This refers to policies designed to lower the carbon intensity of fuels for aviation or to introduce alternative fuel sources, including biofuel blending mandates.", enum=["True", "False"])
    I_Airtraffic: str = Field(description="Measure concerning Air traffic management. Focus on improving air traffic.", enum=["True", "False"])
    I_Shipping: str = Field(description="Measure concerning General shipping improvement. This refers to any general measure that targets shipping, maritime transport or inland navigation. <example> Improved and enhanced Inland Water Transport (IWT) system (Improve navigation for regional, sub-regional, and local routes, improve maintenance of water vessel to enhance engine performance, introduce electric water vessel etc.) </example>", enum=["True", "False"])
    I_Onshorepower: str = Field(description="Measure concerning Support of on-shore power and electric charging facilities in ports. While low-carbon fuels for ships are still being explored, there are already several solutions for providing electricity to vessel when docked. This is also commonly known as “cold ironing”. <example> Highly renewable energy resources will be used in ports. </example>", enum=["True", "False"])
    I_PortInfra: str = Field(description="Measure concerning Port infrastructure improvements. This refers to improvements to ports and other shore-based facilities. <example> promote the development of international marine container terminals and international logistics terminals, and high standardization that incorporates the ICT and IoT technology </example>", enum=["True", "False"])
    I_Shipefficiency: str = Field(description="Measure concerning Ship efficiency improvements. <example> By 2023, quantify the national GHG reductions possible by swifting to lower carbon international maritime transport (i.e.sails or solar vessels, engine efficiency improvement, lower carbon fuels, optimise logistics and operating processes or avoidance strategies etc.) </example>", enum=["True", "False"])

class AdaptationObject_FewShot(BaseModel):
    # ADAPTATION MEASURES
    R_Infrares: str = Field(description="Measure concerning efforts to adapt transport infrastructure to climate change impacts CKANand to increase its resilience", enum=["True", "False"])
    R_System: str = Field(description="Measure concerning efforts to adapt transport system and infrastructure to climate change impacts and to increase its resilience", enum=["True", "False"])
    #R_Maintain: str = Field(description="Measure concerning general efforts to repair or maintain transport infrastructure, without reference to climate change adaptation.", enum=["True", "False"])
    R_Risk: str = Field(description="Measure concerning risk assessments or efforts to understand risks and impacts to the transport system (e.g. through modelling).", enum=["True", "False"])
    R_Tech: str = Field(description="Measure concerning efforts to adopt resilient transport technologies (e.g. climate resilient materials for streets or cars).", enum=["True", "False"])
    R_Monitoring: str = Field(description="Measure concerning efforts to adopt monitoring systems, e.g. to detect risks early on.", enum=["True", "False"])
    #R_Inform: str = Field(description="Measure concerning efforts to adopt notification systems, e.g. to inform drivers about flooding, so they can take alternate routes.", enum=["True", "False"])
    #R_Emergency: str = Field(description="Measure concerning emergency and disaster planning that is specifically related to transport.", enum=["True", "False"])
    R_Education: str = Field(description="Measure concerning efforts to educate and train transport officials regarding the vulnerability of transport systems and infrastructure to climate change.", enum=["True", "False"])
    R_Warning: str = Field(description="Measure containing explicit mention of an early warning system.", enum=["True", "False"])
    R_Planning: str = Field(description="Measure concerning activities designed to raise the importance of resilience and adaptation in transport planning.", enum=["True", "False"])
    R_Relocation: str = Field(description="Measure concerning efforts to relocate infrastructure or populations due to current or anticipated threats.", enum=["True", "False"])
    R_Redundancy: str = Field(description="Measure concerning construction of redundant infrastructure/facilities, to prepare for the possible failure of existing systems. <example>Diversification of transport modes with appropriate adaptive capacities.</example>", enum=["True", "False"])
    #R_Disinvest: str = Field(description="Measure concerning measures to discontinue or avoid expanding transport services or infrastructure. Abandonment or disinvestment of infrastructure", enum=["True", "False"])
    R_Laws: str = Field(description="Measure concerning laws, programmes or regulations that focus on climate change adaptation in the transport sector.", enum=["True", "False"])
    R_Design: str = Field(description="Measure concerning the adoption of improved, more resilient design standards to effectively protect or reinforce transport facilities or infrastructure.", enum=["True", "False"])
    R_Other: str = Field(description="Other adaptation measures for transport not falling under the categories listed above (Structural and Technical, Informational and Educational, Institutional and Regulatory). <example>Produce a research-analysis-assessment platform on climate change risks with impact on transport infrastructure, involving insurance companies</example>", enum=["True", "False"])


def zero_shot_tagging_chain(llm, ResultObject):
    """
    Defines the chain for zero-shot tagging, assigning the ResultObject to the structured Output.

    Parameters
    ----------
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.
    ResultObject : BaseModel
        pydantic BaseModel containing the required fields for tagging/classification

    Returns
    -------
    quote_tagging_chain : langchain_core.runnables.base.RunnableSequence
        Chain object for tagging task.

    """
    
    tagging_prompt = ChatPromptTemplate.from_template(
        """
        Extract the desired information from the following quote.
        
        Only extract the properties mentioned in the 'ResultObject' function.
        
        Quote:
        {quote}
        """
    )
        
    quote_tagging_chain = tagging_prompt | llm.with_structured_output(ResultObject)
    
    return quote_tagging_chain


def zero_shot_tagging_chain_with_context(llm, ResultObject):
    """
    Defines the chain for zero-shot tagging, assigning the ResultObject to the structured Output.

    Parameters
    ----------
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.
    ResultObject : BaseModel
        pydantic BaseModel containing the required fields for tagging/classification

    Returns
    -------
    quote_tagging_chain : langchain_core.runnables.base.RunnableSequence
        Chain object for tagging task.

    """
    
    tagging_prompt = ChatPromptTemplate.from_template(
        """
        Extract the desired information from the following quote, given the context of the quote.
        
        Only extract the properties mentioned in the 'ResultObject' function.
        
        Quote:
        {quote}
        
        Context:
        {context}
        """
    )
        
    quote_tagging_chain = tagging_prompt | llm.with_structured_output(ResultObject)
    
    return quote_tagging_chain


@traceable(
    run_type="chain",
    task="classification",
    version="task decomposition, zero-shot tagging"
    )
def zero_shot_tagger(quote, llm, ResultObject):
    """
    Invokes zero-shot tagging chain to assign tags to provided quote.

    Parameters
    ----------
    quote : str
        Quote to be classified by tagging.
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.
    ResultObject : BaseModel
        pydantic BaseModel containing the required fields for tagging/classification

    Returns
    -------
    res : ClassificationResultObject
        BaseModel object containing the assign tags for each parameter.

    """
    chain = zero_shot_tagging_chain(llm, ResultObject)
    res = chain.invoke({"quote": quote})
    
    return res


@traceable(
    run_type="chain",
    task="classification",
    version="task decomposition, zero-shot tagging"
    )
def zero_shot_tagger_with_context(quote, context, llm, ResultObject):
    """
    Invokes zero-shot tagging chain to assign tags to provided quote.

    Parameters
    ----------
    quote : str
        Quote to be classified by tagging.
    context : str
        Context of the quote to be classified
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.
    ResultObject : BaseModel
        pydantic BaseModel containing the required fields for tagging/classification

    Returns
    -------
    res : ClassificationResultObject
        BaseModel object containing the assign tags for each parameter.

    """
    chain = zero_shot_tagging_chain_with_context(llm, ResultObject)
    res = chain.invoke({"quote": quote, "context" : context})
    
    return res


def get_tagging_results(quotes, llm):
    """
    Applies the tagging classification on a single quote. Returns the results divided into different DataFrames

    Parameters
    ----------
    quotes : pd.Series or List of str
        Strings containing the quotes to be classified.

    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction

    Returns
    -------
    type_df : pd.DataFrame
        Containing results w.r.t. quote type.
    targets_df : pd.DataFrame
        Containing results w.r.t. quote target attributes.
    mitigation_df : pd.DataFrame
        Containing results w.r.t. quote mitigation measures.
    adaptation_df : pd.DataFrame
        Containing results w.r.t. quote adaptation measures.

    """
    
    type_df = pd.DataFrame(columns=['target', 'measure'])
    
    mtype_df = pd.DataFrame(columns=['mitigation_measure', 'adaptation_measure'])

    targets_df = pd.DataFrame(columns=[
                                    'energy', 
                                    'transport', 
                                    'economy_wide', 
                                    'mitigation', 
                                    'adaptation', 
                                    'ghg', 
                                    'net_zero', 
                                    'conditional', 
                                    'unconditional'])
    mitigation_df_TSI = pd.DataFrame(columns=[
                                    'A_Complan', 
                                    'A_Natmobplan', 
                                    'A_SUMP', 
                                    'A_LATM', 
                                    'A_Landuse', 
                                    'A_Density', 
                                    'A_Mixuse', 
                                    'S_Infraimprove', 
                                    'S_Infraexpansion', 
                                    'S_Intermodality', 
                                    'I_Freighteff',
                                    'I_Load',
                                    'S_Railfreight',
                                    'I_Education',
                                    'I_Ecodriving',
                                    'I_Capacity',
                                    'I_Campaigns'
                                    ])
    mitigation_df_MSDM = pd.DataFrame(columns=[
                                    'A_TDM',
                                    'S_Parking',
                                    'A_Parkingprice',
                                    'A_Caraccess',
                                    'A_Commute',
                                    'A_Work',
                                    'A_Teleworking',
                                    'A_Economic',
                                    'A_Emistrad',
                                    'A_Finance',
                                    'A_Procurement',
                                    'A_Fossilfuelsubs',
                                    'A_Fueltax',
                                    'A_Vehicletax',
                                    'A_Roadcharging',
                                    'S_PublicTransport',
                                    'S_PTIntegration',
                                    'S_PTPriority',
                                    'S_BRT',
                                    'S_Activemobility',
                                    'S_Walking',
                                    'S_Cycling',
                                    'S_Sharedmob',
                                    'S_Ondemand',
                                    'S_Maas',
                                    'I_Other',
                                    'I_ITS',
                                    'I_Autonomous',
                                    'I_DataModelling'
                                    ])
    mitigation_df_LCF = pd.DataFrame(columns=[
                                    'I_Vehicleimprove',
                                    'I_Fuelqualimprove',
                                    'I_Inspection',
                                    'I_Efficiencystd',
                                    'I_Vehicleeff',
                                    'A_LEZ',
                                    'I_VehicleRestrictions',
                                    'I_Vehiclescrappage',
                                    'I_Lowemissionincentive',
                                    'I_Altfuels',
                                    'I_Ethanol',
                                    'I_Biofuel',
                                    'I_LPGCNGLNG',
                                    'I_Hydrogen',
                                    'I_RE',
                                    'I_Transportlabel',
                                    'I_Efficiencylabel',
                                    'I_Freightlabel',
                                    'I_Vehiclelabel',
                                    'I_Fuellabel'
                                    ])
    mitigation_df_EI = pd.DataFrame(columns=[
                                    'I_Emobility',
                                    'I_Emobilitycharging',
                                    'I_Smartcharging',
                                    'I_Emobilitypurchase',
                                    'I_ICEdiesel',
                                    'S_Micromobility',
                                    'I_Aviation',
                                    'I_Aircraftfleet',
                                    'I_CO2certificate',
                                    'I_Capacityairport',
                                    'I_Jetfuel',
                                    'I_Airtraffic',
                                    'I_Shipping',
                                    'I_Onshorepower',
                                    'I_PortInfra',
                                    'I_Shipefficiency'
                                    ])
    
    adaptation_df = pd.DataFrame(columns=[
                                               'R_System',
                                               'R_Maintain',
                                               'R_Risk',
                                               'R_Tech',
                                               'R_Monitoring',
                                               'R_Inform',
                                               'R_Emergency',
                                               'R_Education',
                                               'R_Warning',
                                               'R_Planning',
                                               'R_Relocation',
                                               'R_Redundancy',
                                               'R_Disinvest',
                                               'R_Laws',
                                               'R_Design',
                                               'R_Other'
                                               ]
                                        )
    # iterate through quotes
    for quote in quotes:
        # invoke tagger and capture results in tags_df
        tags = zero_shot_tagger(quote, llm, QuoteTypeObject)
        type_df.loc[len(type_df.index)]  = tags.dict()
        
        # tag targets
        if tags.target == 'True':
            
            try:
                t_tags = zero_shot_tagger(quote, llm, TargetObject)
                targets_df.loc[len(targets_df.index)]  = t_tags.dict()
            except Exception:
                # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                # TODO: Retry with GPT 4 or different model
                targets_df.loc[len(targets_df.index)] = 'Error'
                logger.exception('Tagging Error')
            
        else:
            targets_df.loc[len(targets_df.index)] = 'None'
        
        # tag measures
        if tags.measure == 'True':
            
            # invoke tagger and capture results in tags_df
            mt_tags = zero_shot_tagger(quote, llm, MeasureTypeObject)
            mtype_df.loc[len(mtype_df.index)]  = mt_tags.dict()
            
            # tag mitigation
            if mt_tags.mitigation_measure == 'True':
                try:
                    m_tags_1 = zero_shot_tagger(quote, llm, MitigationObject_TSI)
                    mitigation_df_TSI.loc[len(mitigation_df_TSI.index)]  = m_tags_1.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_TSI.loc[len(mitigation_df_TSI.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                try:
                    m_tags_2 = zero_shot_tagger(quote, llm, MitigationObject_MSDM)
                    mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)]  = m_tags_2.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                try:
                    m_tags_3 = zero_shot_tagger(quote, llm, MitigationObject_LCF)
                    mitigation_df_LCF.loc[len(mitigation_df_LCF.index)]  = m_tags_3.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_LCF.loc[len(mitigation_df_LCF.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                try:
                    m_tags_4 = zero_shot_tagger(quote, llm, MitigationObject_EI)
                    mitigation_df_EI.loc[len(mitigation_df_EI.index)]  = m_tags_4.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_EI.loc[len(mitigation_df_EI.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                
            else:
                mitigation_df_TSI.loc[len(mitigation_df_TSI.index)] = 'None'
                mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)] = 'None'
                mitigation_df_LCF.loc[len(mitigation_df_LCF.index)] = 'None'
                mitigation_df_EI.loc[len(mitigation_df_EI.index)] = 'None'
            
            # tag adaptation
            if mt_tags.adaptation_measure == 'True':
                try:
                    a_tags = zero_shot_tagger(quote, llm, AdaptationObject)
                    adaptation_df.loc[len(adaptation_df.index)]  = a_tags.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    adaptation_df.loc[len(adaptation_df.index)] = 'Error'
                    logger.exception('Tagging Error')
                    print(e)
            else:
                adaptation_df.loc[len(adaptation_df.index)] = 'None'
                    
        else:
            mtype_df.loc[len(mtype_df.index)] = 'None'
            mitigation_df_TSI.loc[len(mitigation_df_TSI.index)] = 'None'
            mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)] = 'None'
            mitigation_df_LCF.loc[len(mitigation_df_LCF.index)] = 'None'
            mitigation_df_EI.loc[len(mitigation_df_EI.index)] = 'None'
            adaptation_df.loc[len(adaptation_df.index)] = 'None'
    
    # concat results for each mitigation category
    mitigation_df = pd.concat([mitigation_df_TSI, mitigation_df_MSDM, mitigation_df_LCF, mitigation_df_EI], axis=1) 
        
    return type_df, targets_df, mtype_df, mitigation_df, adaptation_df

def get_tagging_results_fewshot(quotes, llm):
    """
    Applies the tagging classification on a single quot, providing examples. Returns the results divided into different DataFrames

    Parameters
    ----------
    quotes : pd.Series or List of str
        Strings containing the quotes to be classified.

    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction

    Returns
    -------
    type_df : pd.DataFrame
        Containing results w.r.t. quote type.
    targets_df : pd.DataFrame
        Containing results w.r.t. quote target attributes.
    mitigation_df : pd.DataFrame
        Containing results w.r.t. quote mitigation measures.
    adaptation_df : pd.DataFrame
        Containing results w.r.t. quote adaptation measures.

    """
    
    type_df = pd.DataFrame(columns=['target', 'measure'])
    
    mtype_df = pd.DataFrame(columns=['mitigation_measure', 'adaptation_measure'])

    targets_df = pd.DataFrame(columns=[
                                    'energy', 
                                    'transport', 
                                    'economy_wide', 
                                    'mitigation', 
                                    'adaptation', 
                                    'ghg', 
                                    'net_zero', 
                                    'conditional', 
                                    'unconditional'])
    mitigation_df_TSI = pd.DataFrame(columns=[
                                    'A_Complan', 
                                    'A_Natmobplan', 
                                    'A_SUMP', 
                                    'A_LATM', 
                                    'A_Landuse', 
                                    'A_Density', 
                                    'A_Mixuse', 
                                    'S_Infraimprove', 
                                    'S_Infraexpansion', 
                                    'S_Intermodality', 
                                    'I_Freighteff',
                                    'I_Load',
                                    'S_Railfreight',
                                    'I_Education',
                                    'I_Ecodriving',
                                    'I_Capacity',
                                    'I_Campaigns'
                                    ])
    mitigation_df_MSDM = pd.DataFrame(columns=[
                                    'A_TDM',
                                    'S_Parking',
                                    'A_Parkingprice',
                                    'A_Caraccess',
                                    'A_Commute',
                                    'A_Work',
                                    'A_Teleworking',
                                    'A_Economic',
                                    'A_Emistrad',
                                    'A_Finance',
                                    'A_Procurement',
                                    'A_Fossilfuelsubs',
                                    'A_Fueltax',
                                    'A_Vehicletax',
                                    'A_Roadcharging',
                                    'S_PublicTransport',
                                    'S_PTIntegration',
                                    'S_PTPriority',
                                    'S_BRT',
                                    'S_Activemobility',
                                    'S_Walking',
                                    'S_Cycling',
                                    'S_Sharedmob',
                                    'S_Ondemand',
                                    'S_Maas',
                                    'I_Other',
                                    'I_ITS',
                                    'I_Autonomous',
                                    'I_DataModelling'
                                    ])
    mitigation_df_LCF = pd.DataFrame(columns=[
                                    'I_Vehicleimprove',
                                    'I_Fuelqualimprove',
                                    'I_Inspection',
                                    'I_Efficiencystd',
                                    'I_Vehicleeff',
                                    'A_LEZ',
                                    'I_VehicleRestrictions',
                                    'I_Vehiclescrappage',
                                    'I_Lowemissionincentive',
                                    'I_Altfuels',
                                    'I_Ethanol',
                                    'I_Biofuel',
                                    'I_LPGCNGLNG',
                                    'I_Hydrogen',
                                    'I_RE',
                                    'I_Transportlabel',
                                    'I_Efficiencylabel',
                                    'I_Freightlabel',
                                    'I_Vehiclelabel',
                                    'I_Fuellabel'
                                    ])
    mitigation_df_EI = pd.DataFrame(columns=[
                                    'I_Emobility',
                                    'I_Emobilitycharging',
                                    'I_Smartcharging',
                                    'I_Emobilitypurchase',
                                    'I_ICEdiesel',
                                    'S_Micromobility',
                                    'I_Aviation',
                                    'I_Aircraftfleet',
                                    'I_CO2certificate',
                                    'I_Capacityairport',
                                    'I_Jetfuel',
                                    'I_Airtraffic',
                                    'I_Shipping',
                                    'I_Onshorepower',
                                    'I_PortInfra',
                                    'I_Shipefficiency'
                                    ])
    
    adaptation_df = pd.DataFrame(columns=[
                                               'R_System',
                                               'R_Maintain',
                                               'R_Risk',
                                               'R_Tech',
                                               'R_Monitoring',
                                               'R_Inform',
                                               'R_Emergency',
                                               'R_Education',
                                               'R_Warning',
                                               'R_Planning',
                                               'R_Relocation',
                                               'R_Redundancy',
                                               'R_Disinvest',
                                               'R_Laws',
                                               'R_Design',
                                               'R_Other'
                                               ]
                                        )
    # iterate through quotes
    for quote in quotes:
        # invoke tagger and capture results in tags_df
        tags = zero_shot_tagger(quote, llm, QuoteTypeObject)
        type_df.loc[len(type_df.index)]  = tags.dict()
        
        # tag targets
        if tags.target == 'True':
            
            try:
                t_tags = zero_shot_tagger(quote, llm, TargetObject)
                targets_df.loc[len(targets_df.index)]  = t_tags.dict()
            except Exception:
                # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                # TODO: Retry with GPT 4 or different model
                targets_df.loc[len(targets_df.index)] = 'Error'
                logger.exception('Tagging Error')
            
        else:
            targets_df.loc[len(targets_df.index)] = 'None'
        
        # tag measures
        if tags.measure == 'True':
            
            # invoke tagger and capture results in tags_df
            mt_tags = zero_shot_tagger(quote, llm, MeasureTypeObject)
            mtype_df.loc[len(mtype_df.index)]  = mt_tags.dict()
            
            # tag mitigation
            if mt_tags.mitigation_measure == 'True':
                try:
                    m_tags_1 = zero_shot_tagger(quote, llm, MitigationObject_TSI_FewShot)
                    mitigation_df_TSI.loc[len(mitigation_df_TSI.index)]  = m_tags_1.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_TSI.loc[len(mitigation_df_TSI.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                try:
                    m_tags_2 = zero_shot_tagger(quote, llm, MitigationObject_MSDM_FewShot)
                    mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)]  = m_tags_2.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                try:
                    m_tags_3 = zero_shot_tagger(quote, llm, MitigationObject_LCF_FewShot)
                    mitigation_df_LCF.loc[len(mitigation_df_LCF.index)]  = m_tags_3.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_LCF.loc[len(mitigation_df_LCF.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                try:
                    m_tags_4 = zero_shot_tagger(quote, llm, MitigationObject_EI_FewShot)
                    mitigation_df_EI.loc[len(mitigation_df_EI.index)]  = m_tags_4.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_EI.loc[len(mitigation_df_EI.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                
            else:
                mitigation_df_TSI.loc[len(mitigation_df_TSI.index)] = 'None'
                mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)] = 'None'
                mitigation_df_LCF.loc[len(mitigation_df_LCF.index)] = 'None'
                mitigation_df_EI.loc[len(mitigation_df_EI.index)] = 'None'
            
            # tag adaptation
            if mt_tags.adaptation_measure == 'True':
                try:
                    a_tags = zero_shot_tagger(quote, llm, AdaptationObject_FewShot)
                    adaptation_df.loc[len(adaptation_df.index)]  = a_tags.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    adaptation_df.loc[len(adaptation_df.index)] = 'Error'
                    logger.exception('Tagging Error')
                    print(e)
            else:
                adaptation_df.loc[len(adaptation_df.index)] = 'None'
                    
        else:
            mtype_df.loc[len(mtype_df.index)] = 'None'
            mitigation_df_TSI.loc[len(mitigation_df_TSI.index)] = 'None'
            mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)] = 'None'
            mitigation_df_LCF.loc[len(mitigation_df_LCF.index)] = 'None'
            mitigation_df_EI.loc[len(mitigation_df_EI.index)] = 'None'
            adaptation_df.loc[len(adaptation_df.index)] = 'None'
    
    # concat results for each mitigation category
    mitigation_df = pd.concat([mitigation_df_TSI, mitigation_df_MSDM, mitigation_df_LCF, mitigation_df_EI], axis=1) 
        
    return type_df, targets_df, mtype_df, mitigation_df, adaptation_df

def get_tagging_results_with_context(quotes, contexts, llm):
    """
    Applies the tagging classification on a single quote. Returns the results divided into different DataFrames

    Parameters
    ----------
    quotes : pd.Series or List of str
        Strings containing the quotes to be classified.
        
    contexts : pd.Series or List of str
        Strings containing the contexts of the quotes to be classified.

    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction

    Returns
    -------
    type_df : pd.DataFrame
        Containing results w.r.t. quote type.
    targets_df : pd.DataFrame
        Containing results w.r.t. quote target attributes.
    mitigation_df : pd.DataFrame
        Containing results w.r.t. quote mitigation measures.
    adaptation_df : pd.DataFrame
        Containing results w.r.t. quote adaptation measures.

    """
    
    type_df = pd.DataFrame(columns=['target', 'measure'])
    
    mtype_df = pd.DataFrame(columns=['mitigation_measure', 'adaptation_measure'])

    targets_df = pd.DataFrame(columns=[
                                    'energy', 
                                    'transport', 
                                    'economy_wide', 
                                    'mitigation', 
                                    'adaptation', 
                                    'ghg', 
                                    'net_zero', 
                                    'conditional', 
                                    'unconditional'])
    mitigation_df_TSI = pd.DataFrame(columns=[
                                    'A_Complan', 
                                    'A_Natmobplan', 
                                    'A_SUMP', 
                                    'A_LATM', 
                                    'A_Landuse', 
                                    'A_Density', 
                                    'A_Mixuse', 
                                    'S_Infraimprove', 
                                    'S_Infraexpansion', 
                                    'S_Intermodality', 
                                    'I_Freighteff',
                                    'I_Load',
                                    'S_Railfreight',
                                    'I_Education',
                                    'I_Ecodriving',
                                    'I_Capacity',
                                    'I_Campaigns'
                                    ])
    mitigation_df_MSDM = pd.DataFrame(columns=[
                                    'A_TDM',
                                    'S_Parking',
                                    'A_Parkingprice',
                                    'A_Caraccess',
                                    'A_Commute',
                                    'A_Work',
                                    'A_Teleworking',
                                    'A_Economic',
                                    'A_Emistrad',
                                    'A_Finance',
                                    'A_Procurement',
                                    'A_Fossilfuelsubs',
                                    'A_Fueltax',
                                    'A_Vehicletax',
                                    'A_Roadcharging',
                                    'S_PublicTransport',
                                    'S_PTIntegration',
                                    'S_PTPriority',
                                    'S_BRT',
                                    'S_Activemobility',
                                    'S_Walking',
                                    'S_Cycling',
                                    'S_Sharedmob',
                                    'S_Ondemand',
                                    'S_Maas',
                                    'I_Other',
                                    'I_ITS',
                                    'I_Autonomous',
                                    'I_DataModelling'
                                    ])
    mitigation_df_LCF = pd.DataFrame(columns=[
                                    'I_Vehicleimprove',
                                    'I_Fuelqualimprove',
                                    'I_Inspection',
                                    'I_Efficiencystd',
                                    'I_Vehicleeff',
                                    'A_LEZ',
                                    'I_VehicleRestrictions',
                                    'I_Vehiclescrappage',
                                    'I_Lowemissionincentive',
                                    'I_Altfuels',
                                    'I_Ethanol',
                                    'I_Biofuel',
                                    'I_LPGCNGLNG',
                                    'I_Hydrogen',
                                    'I_RE',
                                    'I_Transportlabel',
                                    'I_Efficiencylabel',
                                    'I_Freightlabel',
                                    'I_Vehiclelabel',
                                    'I_Fuellabel'
                                    ])
    mitigation_df_EI = pd.DataFrame(columns=[
                                    'I_Emobility',
                                    'I_Emobilitycharging',
                                    'I_Smartcharging',
                                    'I_Emobilitypurchase',
                                    'I_ICEdiesel',
                                    'S_Micromobility',
                                    'I_Aviation',
                                    'I_Aircraftfleet',
                                    'I_CO2certificate',
                                    'I_Capacityairport',
                                    'I_Jetfuel',
                                    'I_Airtraffic',
                                    'I_Shipping',
                                    'I_Onshorepower',
                                    'I_PortInfra',
                                    'I_Shipefficiency'
                                    ])
    
    adaptation_df = pd.DataFrame(columns=[
                                               'R_System',
                                               'R_Maintain',
                                               'R_Risk',
                                               'R_Tech',
                                               'R_Monitoring',
                                               'R_Inform',
                                               'R_Emergency',
                                               'R_Education',
                                               'R_Warning',
                                               'R_Planning',
                                               'R_Relocation',
                                               'R_Redundancy',
                                               'R_Disinvest',
                                               'R_Laws',
                                               'R_Design',
                                               'R_Other'
                                               ]
                                        )
    # iterate through quotes
    for i, quote in enumerate(quotes):
        # invoke tagger and capture results in tags_df
        tags = zero_shot_tagger_with_context(quote, contexts[i], llm, QuoteTypeObject)
        type_df.loc[len(type_df.index)]  = tags.dict()
        
        # tag targets
        if tags.target == 'True':
            
            try:
                t_tags = zero_shot_tagger_with_context(quote, contexts[i], llm, TargetObject)
                targets_df.loc[len(targets_df.index)]  = t_tags.dict()
            except Exception:
                # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                # TODO: Retry with GPT 4 or different model
                targets_df.loc[len(targets_df.index)] = 'Error'
                logger.exception('Tagging Error')
            
        else:
            targets_df.loc[len(targets_df.index)] = 'None'
        
        # tag measures
        if tags.measure == 'True':
            
            # invoke tagger and capture results in tags_df
            mt_tags = zero_shot_tagger_with_context(quote, contexts[i], llm, MeasureTypeObject)
            mtype_df.loc[len(mtype_df.index)]  = mt_tags.dict()
            
            # tag mitigation
            if mt_tags.mitigation_measure == 'True':
                try:
                    m_tags_1 = zero_shot_tagger_with_context(quote, contexts[i], llm, MitigationObject_TSI)
                    mitigation_df_TSI.loc[len(mitigation_df_TSI.index)]  = m_tags_1.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_TSI.loc[len(mitigation_df_TSI.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                try:
                    m_tags_2 = zero_shot_tagger_with_context(quote, contexts[i], llm, MitigationObject_MSDM)
                    mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)]  = m_tags_2.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                try:
                    m_tags_3 = zero_shot_tagger_with_context(quote, contexts[i], llm, MitigationObject_LCF)
                    mitigation_df_LCF.loc[len(mitigation_df_LCF.index)]  = m_tags_3.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_LCF.loc[len(mitigation_df_LCF.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                try:
                    m_tags_4 = zero_shot_tagger_with_context(quote, contexts[i], llm, MitigationObject_EI)
                    mitigation_df_EI.loc[len(mitigation_df_EI.index)]  = m_tags_4.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    mitigation_df_EI.loc[len(mitigation_df_EI.index)] = 'Error'
                    print(e)
                    logger.exception('Tagging Error')
                
            else:
                mitigation_df_TSI.loc[len(mitigation_df_TSI.index)] = 'None'
                mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)] = 'None'
                mitigation_df_LCF.loc[len(mitigation_df_LCF.index)] = 'None'
                mitigation_df_EI.loc[len(mitigation_df_EI.index)] = 'None'
            
            # tag adaptation
            if mt_tags.adaptation_measure == 'True':
                try:
                    a_tags = zero_shot_tagger_with_context(quote, contexts[i], llm, AdaptationObject)
                    adaptation_df.loc[len(adaptation_df.index)]  = a_tags.dict()
                except Exception as e:
                    # https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#tryexcept-tool-call
                    # TODO: Retry with GPT 4 or different model
                    adaptation_df.loc[len(adaptation_df.index)] = 'Error'
                    logger.exception('Tagging Error')
                    print(e)
            else:
                adaptation_df.loc[len(adaptation_df.index)] = 'None'
                    
        else:
            mtype_df.loc[len(mtype_df.index)] = 'None'
            mitigation_df_TSI.loc[len(mitigation_df_TSI.index)] = 'None'
            mitigation_df_MSDM.loc[len(mitigation_df_MSDM.index)] = 'None'
            mitigation_df_LCF.loc[len(mitigation_df_LCF.index)] = 'None'
            mitigation_df_EI.loc[len(mitigation_df_EI.index)] = 'None'
            adaptation_df.loc[len(adaptation_df.index)] = 'None'
    
    # concat results for each mitigation category
    mitigation_df = pd.concat([mitigation_df_TSI, mitigation_df_MSDM, mitigation_df_LCF, mitigation_df_EI], axis=1) 
        
    return type_df, targets_df, mtype_df, mitigation_df, adaptation_df


def tagging_classifier_quotes(quotes_dict, llm, fewshot = True):
    """
    Applies tagging-based classification for the retrieved quotes:
       -  https://python.langchain.com/v0.1/docs/use_cases/tagging/
       -  https://python.langchain.com/v0.1/docs/integrations/document_transformers/openai_metadata_tagger/

    Parameters
    ----------
    quotes_dict : dict
        Dictionary containing a list of quotes and metadata for each retrieved document.

    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.
    
    fewshot : bool
        Selection between Zero-Shot (False) and Few-Shot (True). Default: True
        
    Returns
    -------
    output_df : pd.DataFrame
        DataFrame containing the complete results from classification tagging.

    """
    # initialize output dataframes
    output_df = pd.DataFrame(columns=['document', 'page', 'info_type', 'quote', 'target_labels', 'measure_labels'])
    
    for key, value in quotes_dict.items():
        
        if type(value) == str:
            continue

        # check if quotes available
        if len(value['quotes'])>0:
            
            for q in value['quotes']:            
                #create entry for each quote
                entry = []
                entry.append(value['filename'])
                entry.append(int(value['page_number']))
                entry.append(value['type'])
                #entry.append(str(value['keywords']))
                entry.append(q[1])
                entry.append('NONE') #placeholder for target labels
                entry.append('NONE') #placeholder for measure labels
                output_df.loc[len(output_df.index)] = entry
                
    if fewshot:
        type_df, targets_df, mtype_df, mitigation_df, adaptation_df = get_tagging_results_fewshot(quotes=output_df.quote, llm=llm)
    else:      
        type_df, targets_df, mtype_df, mitigation_df, adaptation_df = get_tagging_results(quotes=output_df.quote, llm=llm)
                
    # concatenate both DataFrames
    output_df = pd.concat([output_df, type_df, targets_df, mtype_df, mitigation_df, adaptation_df], axis=1)        
    
    # mapping
    try:
        output_df['target_labels'] = output_df.apply(lambda row : target_mapping_new(row), axis=1)
        output_df['measure_labels'] = output_df.apply(lambda row : measure_mapping(row), axis=1)
    except Exception:
        logger.exception('Error while applying label mappers')
    return output_df



def type_mapping(row):
    """
    Type label attribution according to the tagging results.

    Parameters
    ----------
    row : pd.Series
        Row of a Dataframe containing the tagging results.

    Returns
    -------
    targets : list
        List of applicable type labels for the specified row.

    """
    #if row['target'] == 'True':
    types = [] 
    if row['target'] == 'True':
        types.append('Target')
    if row['measure'] == 'True':
        types.append('Measure')
        
    return types

def target_mapping(row):
    """
    Target label attribution according to the tagging results.

    Parameters
    ----------
    row : pd.Series
        Row of a Dataframe containing the tagging results.

    Returns
    -------
    targets : list
        List of applicable target labels for the specified row.

    """
    #if row['target'] == 'True':
    targets = [] 
    if row['net_zero'] == 'True':
        targets.append('T_Netzero')
    if (row['energy'] == 'True') & (row['mitigation'] == 'True'):
        targets.append('T_Energy')
    if (row['economy_wide'] == 'True') & (row['mitigation'] == 'True') & (row['ghg'] == 'True') & (row['conditional'] == 'False'):
        targets.append('T_Economy_Unc')
    if (row['economy_wide'] == 'True') & (row['mitigation'] == 'True') & (row['ghg'] == 'True') & (row['conditional'] == 'True'):
        targets.append('T_Economy_C')
    if (row['transport'] == 'True') & (row['mitigation'] == 'True') & (row['ghg'] == 'True') & (row['conditional'] == 'False'):
        targets.append('T_Transport_Unc')
    if (row['transport'] == 'True') & (row['mitigation'] == 'True') & (row['ghg'] == 'True') & (row['conditional'] == 'True'):
        targets.append('T_Transport_C')
    if (row['transport'] == 'True') & (row['adaptation'] == 'True') & (row['ghg'] == 'False') & (row['conditional'] == 'False'):
        targets.append('T_Adaptation_Unc')
    if (row['transport'] == 'True') & (row['adaptation'] == 'True') & (row['ghg'] == 'False') & (row['conditional'] == 'True'):
        targets.append('T_Adaptation_C')
    if (row['transport'] == 'True') & (row['mitigation'] == 'True') & (row['ghg'] == 'False') & (row['conditional'] == 'False'):
        targets.append('T_Transport_O_Unc')
    if (row['transport'] == 'True') & (row['mitigation'] == 'True') & (row['ghg'] == 'False') & (row['conditional'] == 'True'):
        targets.append('T_Transport_O_C')
        
    return targets

def target_mapping_new(row):
    """
    Target label attribution according to the tagging results.

    Parameters
    ----------
    row : pd.Series
        Row of a Dataframe containing the tagging results.

    Returns
    -------
    targets : list
        List of applicable target labels for the specified row.

    """
    #if row['target'] == 'True':
    targets = [] 
    if row['net_zero'] == 'True':
        targets.append('T_Netzero')
    if (row['energy'] == 'True') & (row['mitigation'] == 'True'):
        targets.append('T_Energy')
    if (row['economy_wide'] == 'True') & (row['mitigation'] == 'True') & (row['ghg'] == 'True'):
        targets.append('T_Economy')
    if (row['transport'] == 'True') & (row['mitigation'] == 'True') & (row['ghg'] == 'True'):
        targets.append('T_Transport')
    if (row['transport'] == 'True') & (row['adaptation'] == 'True') & (row['ghg'] == 'False'):
        targets.append('T_Adaptation')
    if (row['transport'] == 'True') & (row['mitigation'] == 'True') & (row['ghg'] == 'False'):
        targets.append('T_NONGHG') #NEW
    if (row['mitigation'] == 'True') & (row['ghg'] == 'True'): 
        targets.append('T_GHG') #NEW
    if (row['conditional'] == 'True') & (row['unconditional'] == 'False'):
        targets.append('T_Conditional') #NEW
    if (row['conditional'] == 'False') & (row['unconditional'] == 'True'):
        targets.append('T_Conditional') #NEW
    if (row['conditional'] == 'False') & (row['unconditional'] == 'False'):
        targets.append('T_Unclear') #NEW
    if (row['conditional'] == 'True') & (row['unconditional'] == 'True'):
        targets.append('T_Unclear') #NEW
        
    return targets

def parameter_type_mapping(row):
    """
    Parameter Type attribution according to the ground truth Parameter.

    Parameters
    ----------
    row : pd.Series
        Row of a Dataframe containing the tagging results.

    Returns
    -------
    targets : list
        List of applicable type labels for the specified row.

    """
    
    mtype : str 
    if row['Parameter'] in ['A_Complan', 'A_Natmobplan', 'A_SUMP', 'A_LATM', 'A_Landuse',
           'A_Density', 'A_Mixuse', 'S_Infraimprove', 'S_Infraexpansion',
           'S_Intermodality', 'I_Freighteff', 'I_Load', 'S_Railfreight',
           'I_Education', 'I_Ecodriving', 'I_Capacity', 'I_Campaigns', 'A_TDM',
           'S_Parking', 'A_Parkingprice', 'A_Caraccess', 'A_Commute', 'A_Work',
           'A_Teleworking', 'A_Economic', 'A_Emistrad', 'A_Finance',
           'A_Procurement', 'A_Fossilfuelsubs', 'A_Fueltax', 'A_Vehicletax',
           'A_Roadcharging', 'S_PublicTransport', 'S_PTIntegration',
           'S_PTPriority', 'S_BRT', 'S_Activemobility', 'S_Walking', 'S_Cycling',
           'S_Sharedmob', 'S_Ondemand', 'S_Maas', 'I_Other', 'I_ITS',
           'I_Autonomous', 'I_DataModelling', 'I_Vehicleimprove',
           'I_Fuelqualimprove', 'I_Inspection', 'I_Efficiencystd', 'I_Vehicleeff',
           'A_LEZ', 'I_VehicleRestrictions', 'I_Vehiclescrappage',
           'I_Lowemissionincentive', 'I_Altfuels', 'I_Ethanol', 'I_Biofuel',
           'I_LPGCNGLNG', 'I_Hydrogen', 'I_RE', 'I_Transportlabel',
           'I_Efficiencylabel', 'I_Freightlabel', 'I_Vehiclelabel', 'I_Fuellabel',
           'I_Emobility', 'I_Emobilitycharging', 'I_Smartcharging',
           'I_Emobilitypurchase', 'I_ICEdiesel', 'S_Micromobility', 'I_Aviation',
           'I_Aircraftfleet', 'I_CO2certificate', 'I_Capacityairport', 'I_Jetfuel',
           'I_Airtraffic', 'I_Shipping', 'I_Onshorepower', 'I_PortInfra',
           'I_Shipefficiency']:
        mtype='Mitigation'
    if row['Parameter'] in ['R_System', 'R_Maintain', 'R_Risk', 'R_Tech', 'R_Monitoring',
           'R_Inform', 'R_Emergency', 'R_Education', 'R_Warning', 'R_Planning',
           'R_Relocation', 'R_Redundancy', 'R_Disinvest', 'R_Laws', 'R_Design',
           'R_Other']:
        mtype='Adaptation'
        
    return mtype

def measure_type_mapping(row):
    """
    Type label attribution according to the tagging results.

    Parameters
    ----------
    row : pd.Series
        Row of a Dataframe containing the tagging results.

    Returns
    -------
    targets : list
        List of applicable type labels for the specified row.

    """
    #if row['target'] == 'True':
    types = [] 
    if row['mitigation_measure'] == 'True':
        types.append('Mitigation')
    if row['adaptation_measure'] == 'True':
        types.append('Adaptation')
        
    return types

def measure_mapping(row):
    """
    Measure label attribution according to the tagging results.

    Parameters
    ----------
    row : pd.Series
        Row of a Dataframe containing the tagging results.

    Returns
    -------
    targets : list
        List of applicable measure labels for the specified row.

    """
    #if row['measure'] == 'True':
    measures = [] 
    # Mitigation
    if row['A_Complan'] == 'True':
        measures.append('A_Complan')
    if row['A_Natmobplan'] == 'True':
        measures.append('A_Natmobplan')
    if row['A_SUMP'] == 'True':
        measures.append('A_SUMP')
    if row['A_LATM'] == 'True':
        measures.append('A_LATM')
    if row['A_Landuse'] == 'True':
        measures.append('A_Landuse')
    if row['A_Density'] == 'True':
        measures.append('A_Density')        
    if row['A_Mixuse'] == 'True':
        measures.append('A_Mixuse')
    if row['S_Infraimprove'] == 'True':
        measures.append('S_Infraimprove')
    if row['S_Infraexpansion'] == 'True':
        measures.append('S_Infraexpansion')        
    if row['S_Intermodality'] == 'True':
        measures.append('S_Intermodality')
    if row['I_Freighteff'] == 'True':
        measures.append('I_Freighteff')
    if row['I_Load'] == 'True':
        measures.append('I_Load')
    if row['S_Railfreight'] == 'True':
        measures.append('S_Railfreight') 
    if row['I_Education'] == 'True':
        measures.append('I_Education')  
    if row['I_Ecodriving'] == 'True':
        measures.append('I_Ecodriving')  
    if row['I_Capacity'] == 'True':
        measures.append('I_Capacity')  
    if row['I_Campaigns'] == 'True':
        measures.append('I_Campaigns')  
    if row['A_TDM'] == 'True':
        measures.append('A_TDM')  
    if row['S_Parking'] == 'True':
        measures.append('S_Parking')  
    if row['A_Parkingprice'] == 'True':
        measures.append('A_Parkingprice')  
    if row['A_Caraccess'] == 'True':
        measures.append('A_Caraccess')  
    if row['A_Commute'] == 'True':
        measures.append('A_Commute')  
    if row['A_Work'] == 'True':
        measures.append('A_Work')  
    if row['A_Teleworking'] == 'True':
        measures.append('A_Teleworking')  
    if row['A_Economic'] == 'True':
        measures.append('A_Economic')       
    if row['A_Emistrad'] == 'True':
        measures.append('A_Emistrad')     
    if row['A_Finance'] == 'True':
        measures.append('A_Finance')     
    if row['A_Procurement'] == 'True':
        measures.append('A_Procurement')     
    if row['A_Fossilfuelsubs'] == 'True':
        measures.append('A_Fossilfuelsubs')     
    if row['A_Fueltax'] == 'True':
        measures.append('A_Fueltax')     
    if row['A_Vehicletax'] == 'True':
        measures.append('A_Vehicletax')     
    if row['A_Roadcharging'] == 'True':
        measures.append('A_Roadcharging')     
    if row['S_PublicTransport'] == 'True':
        measures.append('S_PublicTransport')     
    if row['S_PTIntegration'] == 'True':
        measures.append('S_PTIntegration')       
    if row['S_PTPriority'] == 'True':
        measures.append('S_PTPriority')     
    if row['S_BRT'] == 'True':
        measures.append('S_BRT')     
    if row['S_Activemobility'] == 'True':
        measures.append('S_Activemobility')     
    if row['S_Walking'] == 'True':
        measures.append('S_Walking')     
    if row['S_Cycling'] == 'True':
        measures.append('S_Cycling')     
    if row['S_Sharedmob'] == 'True':
        measures.append('S_Sharedmob')     
    if row['S_Ondemand'] == 'True':
        measures.append('S_Ondemand')     
    if row['S_Maas'] == 'True':
        measures.append('S_Maas')     
    if row['I_Other'] == 'True':
        measures.append('I_Other')     
    if row['I_ITS'] == 'True':
        measures.append('I_ITS')     
    if row['I_Autonomous'] == 'True':
        measures.append('I_Autonomous')     
    if row['I_DataModelling'] == 'True':
        measures.append('I_DataModelling')     
    if row['I_Vehicleimprove'] == 'True':
        measures.append('I_Vehicleimprove')     
    if row['I_Fuelqualimprove'] == 'True':
        measures.append('I_Fuelqualimprove')    
    if row['I_Inspection'] == 'True':
        measures.append('I_Inspection')    
    if row['I_Efficiencystd'] == 'True':
        measures.append('I_Efficiencystd')    
    if row['I_Vehicleeff'] == 'True':
        measures.append('I_Vehicleeff')    
    if row['A_LEZ'] == 'True':
        measures.append('A_LEZ')    
    if row['I_VehicleRestrictions'] == 'True':
        measures.append('I_VehicleRestrictions')    
    if row['I_Vehiclescrappage'] == 'True':
        measures.append('I_Vehiclescrappage')      
    if row['I_Lowemissionincentive'] == 'True':
        measures.append('I_Lowemissionincentive')      
    if row['I_Altfuels'] == 'True':
        measures.append('I_Altfuels')      
    if row['I_Ethanol'] == 'True':
        measures.append('I_Ethanol')      
    if row['I_Biofuel'] == 'True':
        measures.append('I_Biofuel')      
    if row['I_LPGCNGLNG'] == 'True':
        measures.append('I_LPGCNGLNG')     
    if row['I_Hydrogen'] == 'True':
        measures.append('I_Hydrogen')    
    if row['I_RE'] == 'True':
        measures.append('I_RE')    
    if row['I_Transportlabel'] == 'True':
        measures.append('I_Transportlabel')    
    if row['I_Efficiencylabel'] == 'True':
        measures.append('I_Efficiencylabel')    
    if row['I_Freightlabel'] == 'True':
        measures.append('I_Freightlabel')     
    if row['I_Vehiclelabel'] == 'True':
        measures.append('I_Vehiclelabel')   
    if row['I_Freightlabel'] == 'True':
        measures.append('I_Freightlabel')   
    if row['I_Fuellabel'] == 'True':
        measures.append('I_Fuellabel')   
    if row['I_Emobility'] == 'True':
        measures.append('I_Emobility')   
    if row['I_Emobilitycharging'] == 'True':
        measures.append('I_Emobilitycharging')   
    if row['I_Smartcharging'] == 'True':
        measures.append('I_Smartcharging')   
    if row['I_Emobilitypurchase'] == 'True':
        measures.append('I_Emobilitypurchase')   
    if row['I_ICEdiesel'] == 'True':
        measures.append('I_ICEdiesel')   
    if row['S_Micromobility'] == 'True':
        measures.append('S_Micromobility')   
    if row['I_Aviation'] == 'True':
        measures.append('I_Aviation')   
    if row['I_Aircraftfleet'] == 'True':
        measures.append('I_Aircraftfleet') 
    if row['I_CO2certificate'] == 'True':
        measures.append('I_CO2certificate') 
    if row['I_Capacityairport'] == 'True':
        measures.append('I_Capacityairport') 
    if row['I_Jetfuel'] == 'True':
        measures.append('I_Jetfuel') 
    if row['I_Airtraffic'] == 'True':
        measures.append('I_Airtraffic') 
    if row['I_Shipping'] == 'True':
        measures.append('I_Shipping') 
    if row['I_Onshorepower'] == 'True':
        measures.append('I_Onshorepower')  
    if row['I_PortInfra'] == 'True':
        measures.append('I_PortInfra')  
    if row['I_Shipefficiency'] == 'True':
        measures.append('I_Shipefficiency')  
    # I_Other mitigation measure:
    #mitigation_list = ['A_Complan', 'A_Natmobplan', 'A_SUMP', 'A_LATM', 'A_Landuse', 'A_Density', 'A_Mixuse', 'S_Infraimprove', 'S_Infraexpansion', 'S_Intermodality', 'I_Freighteff','I_Load','S_Railfreight','I_Education','I_Ecodriving','I_Capacity','I_Campaigns','A_TDM','S_Parking','A_Parkingprice','A_Caraccess','A_Commute','A_Work','A_Teleworking','A_Economic','A_Emistrad','A_Finance','A_Procurement','A_Fossilfuelsubs','A_Fueltax','A_Vehicletax','A_Roadcharging','S_PublicTransport','S_PTIntegration','S_PTPriority','S_BRT','S_Activemobility','S_Walking','S_Cycling','S_Sharedmob','S_Ondemand','S_Maas','I_Other','I_ITS','I_Autonomous','I_DataModelling','I_Vehicleimprove','I_Fuelqualimprove','I_Inspection','I_Efficiencystd','I_Vehicleeff','A_LEZ','I_VehicleRestrictions','I_Vehiclescrappage','I_Lowemissionincentive','I_Altfuels','I_Ethanol','I_Biofuel','I_LPGCNGLNG','I_Hydrogen','I_RE','I_Transportlabel','I_Efficiencylabel','I_Freightlabel','I_Vehiclelabel','I_Fuellabel','I_Emobility','I_Emobilitycharging','I_Smartcharging','I_Emobilitypurchase','I_ICEdiesel','S_Micromobility','I_Aviation','I_Aircraftfleet','I_CO2certificate', 'I_Capacityairport','I_Jetfuel','I_Airtraffic','I_Shipping','I_Onshorepower','I_PortInfra','I_Shipefficiency']
    #if row[mitigation_list].apply(lambda row: all(row == 'False'), axis=1): #TODO: To be tested
    #    measures.append('I_Other') 
        
    # Adaptation
    if row['R_System'] == 'True':
        measures.append('R_System')  
    if row['R_Maintain'] == 'True':
        measures.append('R_Maintain')  
    if row['R_Risk'] == 'True':
        measures.append('R_Risk')  
    if row['R_Tech'] == 'True':
        measures.append('R_Tech')  
    if row['R_Monitoring'] == 'True':
        measures.append('R_Monitoring')  
    if row['R_Inform'] == 'True':
        measures.append('R_Inform')  
    if row['R_Emergency'] == 'True':
        measures.append('R_Emergency') 
    if row['R_Education'] == 'True':
        measures.append('R_Education')  
    if row['R_Warning'] == 'True':
        measures.append('R_Warning')  
    if row['R_Planning'] == 'True':
        measures.append('R_Planning')  
    if row['R_Relocation'] == 'True':
        measures.append('R_Relocation')   
    if row['R_Redundancy'] == 'True':
        measures.append('R_Redundancy')   
    if row['R_Disinvest'] == 'True':
        measures.append('R_Disinvest')   
    if row['R_Laws'] == 'True':
        measures.append('R_Laws')   
    if row['R_Design'] == 'True':
        measures.append('R_Design')   
    if row['R_Other'] == 'True':
        measures.append('R_Other')  
    # R_Other adaptation measure:
    #adaptation_list = ['R_System', 'R_Maintain', 'R_Risk', 'R_Tech', 'R_Monitoring', 'R_Inform', 'R_Emergency', 'R_Education', 'R_Warning', 'R_Planning', 'R_Relocation','R_Redundancy','R_Disinvest','R_Laws','R_Design']
    #if row[adaptation_list].apply(lambda row: all(row == 'False'), axis=1): #TODO: To be tested
    #    measures.append('R_Other')
        
    return measures
