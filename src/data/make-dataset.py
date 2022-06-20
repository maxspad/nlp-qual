# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.9.13 ('nlp-qual-max')
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

proj_path = '../../'
comment_path = proj_path + 'data/raw/comments/'
qual_ratings_path = proj_path + 'data/raw/qual-ratings/'

# +
# SASK 
# comment column name: Feedback

sasdf = pd.read_excel(comment_path + 'sask-database-de-identified-comments.xlsx')

SasCommentCol = "Feedback"
SasEPACol = "EPA"
SasRatingCol = 'Rating'
SasRawNumber = 'Random Number'
SasObserverType = 'Observer Type'
SasObserverSpecialty = 'EM/PEM vs off-service'


# MAC 
# comment column name: Please provide comments about how the resident performed on THIS specific EPA.

macdf = pd.read_excel(comment_path + 'mcmaster-database-de-identified-comments.xlsx')

MacCommentCol = "Please provide comments about how the resident performed on THIS specific EPA."
MacEPACol = "EPA"
MacRatingCol = "Based on my observation of the resident's performance on this EPA today:"
MacRawNumber = 'Number'
MacGenderRes = 'GenderRes'
MacGenderFac = 'GenderFac'
MacObserverType = 'Unnamed: 6'
MacObserverSpecialty = 'Type'
MacPGY = 'PGY'

dataPath = qual_ratings_path # path for raw excel files

q1 = "Does the rater provide sufficient evidence about performance?"
q2 = "Does the rater provide a suggestion for improvement?"
q3 = "Is the rater's suggestion linked to the behaviour described?"

isResidentQuestion = "Are you a resident or an attending physician?"
# -

# Final columns
# - commentId
# - dataSource
# - NumberFromRawData
# - EPA
# - GenderRes (Mac only)
# - GenderFac (Mac only)
# - ObserverType
# - ObserverSpecialty 
# - PGY (Mac only)
# - comment
# - rating
# - Survey N
# - Question N
# - q1p1
# - q1p2
# - q2p1
# - q2p2
# - q3p1
# - q3p2
#

# +
currentSurveyNo = 0
currentQuestionNo = 0
# currentData = pd.read_excel(dataPath + "/EPA Narrative Comment Survey v01.xlsx")

masterdb = pd.DataFrame(columns = ["commentId", "dataSource", "NumberFromRawData", "comment", "rating", "Survey N", "Question N", "q1p1", "q1p2", "q2p1", "q2p2", "q3p1", "q3p2"])

for index, row in sasdf.iterrows():
    if currentSurveyNo != row['Survey N']:
        currentSurveyNo = row['Survey N']
        if currentSurveyNo < 10:
            surveyNo = "0" + str(currentSurveyNo)
        else:
            surveyNo = str(currentSurveyNo)
        currentData = pd.read_excel(dataPath + "/EPA Narrative Comment Survey v"+ surveyNo + ".xlsx").drop(0).sort_values(isResidentQuestion) #Sor the residency question so that participant 1 in the masterdb answers as resident and the participant 2 is attending.

    currentQuestionNo = row['Question N']
    if currentQuestionNo == 1:
        questionNo = ""
    else:
        questionNo = "." + str(row['Question N'] - 1) #df column name determination
    
    q1Col = q1 + questionNo
    
#     print(q1Col)
    
    q2Col = q2 + questionNo
    q3Col = q3 + questionNo
    
#     print(currentData.iloc[1, currentData.columns.get_loc(q1Col)])
    
    newrow = {
        "commentId": index,
        "dataSource": 'Sas',
        "NumberFromRawData": row[SasRawNumber],
        'EPA': row[SasEPACol],
        'GenderRes': np.nan,
        'GenderFac': np.nan,
        'ObserverType': row[SasObserverType],
        'ObserverSpecialty': row[SasObserverSpecialty],
        'PGY': np.nan,
        "comment": row[SasCommentCol],
        "rating": row[SasRatingCol],
        "Survey N": row['Survey N'],
        "Question N": row['Question N'],
        "q1p1": currentData.iloc[0, currentData.columns.get_loc(q1Col)],
        "q1p2": currentData.iloc[1, currentData.columns.get_loc(q1Col)],         
        "q2p1": currentData.iloc[0, currentData.columns.get_loc(q2Col)],
        "q2p2": currentData.iloc[1, currentData.columns.get_loc(q2Col)],      
        "q3p1": currentData.iloc[0, currentData.columns.get_loc(q3Col)],
        "q3p2": currentData.iloc[1, currentData.columns.get_loc(q3Col)]
    }
    
#     print(newrow)
    
    masterdb = masterdb.append(
        newrow
        , ignore_index=True)
    
#     break


###start processing mcmaster data ###

for index, row in macdf.iterrows():
    if currentSurveyNo != row['Survey N']:
        currentSurveyNo = row['Survey N']
        if currentSurveyNo < 10:
            surveyNo = "0" + str(currentSurveyNo)
        else:
            surveyNo = str(currentSurveyNo)
        currentData = pd.read_excel(dataPath + "/EPA Narrative Comment Survey v"+ surveyNo + ".xlsx").drop(0)

    currentQuestionNo = row['Question N']
    if currentQuestionNo == 1:
        questionNo = ""
    else:
        questionNo = "." + str(row['Question N'] - 1) #df column name determination
    
    q1Col = q1 + questionNo
    
#     print(q1Col)
    
    q2Col = q2 + questionNo
    q3Col = q3 + questionNo
    
#     print(currentData.iloc[1, currentData.columns.get_loc(q1Col)])
    
    newrow = {
        "commentId": index,
        "dataSource": 'Mac',
        "NumberFromRawData": row[MacRawNumber],
        "EPA": row[MacEPACol].split(':')[0],
        'GenderRes': row[MacGenderRes],
        'GenderFac': row[MacGenderFac],
        'ObserverType': row[MacObserverType],
        'ObserverSpecialty': row[MacObserverSpecialty],
        'PGY': row[MacPGY],
        "comment": row[MacCommentCol],
        "rating": row[MacRatingCol], ##mcmaster EPA rating col
        "Survey N": row['Survey N'],
        "Question N": row['Question N'],
        "q1p1": currentData.iloc[0, currentData.columns.get_loc(q1Col)],
        "q1p2": currentData.iloc[1, currentData.columns.get_loc(q1Col)],         
        "q2p1": currentData.iloc[0, currentData.columns.get_loc(q2Col)],
        "q2p2": currentData.iloc[1, currentData.columns.get_loc(q2Col)],      
        "q3p1": currentData.iloc[0, currentData.columns.get_loc(q3Col)],
        "q3p2": currentData.iloc[1, currentData.columns.get_loc(q3Col)]
    }
    
    masterdb = masterdb.append(newrow, ignore_index=True)
    

masterdb.to_excel("masterdb.xlsx")

print("Done! Data merged!")
    
# print(row['Feedback'], row['Survey N'], row['Question N'])

# +
#### Transform data to fit the QUAL score metrics.


masterdb['q1p1T'] = masterdb['q1p1'].replace([1,2,3,4],[3,2,1,0])
masterdb['q1p2T'] = masterdb['q1p2'].replace([1,2,3,4],[3,2,1,0])

masterdb['q2p1T'] = masterdb['q2p1'].replace([1,2],[1,0])
masterdb['q2p2T'] = masterdb['q2p2'].replace([1,2],[1,0])

masterdb['q3p1T'] = masterdb['q3p1'].replace([1,2],[1,0])
masterdb['q3p2T'] = masterdb['q3p2'].replace([1,2],[1,0])


masterdb.to_excel("masterdb.xlsx")

# +
#calculate. qual scores for each participant


masterdb['P1QualScore'] = masterdb['q1p1T'] + masterdb['q2p1T'] + masterdb['q3p1T']

masterdb['P2QualScore'] = masterdb['q1p2T'] + masterdb['q2p2T'] + masterdb['q3p2T']

masterdb.to_excel("masterdb.xlsx")

# +
#compare participant answer to find perfect match and descripincies.


masterdb['Q1Match'] = np.where(masterdb['q1p1T'] == masterdb['q1p2T'],"TRUE","FALSE")

masterdb['Q2Match'] = np.where(masterdb['q2p1T'] == masterdb['q2p2T'],"TRUE","FALSE")

masterdb['Q3Match'] = np.where(masterdb['q3p1T'] == masterdb['q3p2T'],"TRUE","FALSE")


masterdb['perfectMatch'] = np.where((masterdb['Q1Match'] == "TRUE") & (masterdb['Q2Match'] == "TRUE") & (masterdb['Q3Match'] == "TRUE"),"TRUE","FALSE")


masterdb.to_excel("masterdb.xlsx")

# +
#merge Rob and Mac's scorrings on previous data

RobMacSaskData = pd.read_excel(proj_path + "data/raw/Sask Database with Numerical QuAL Scores.xlsx")

RobMacMacData = pd.read_excel(proj_path + "data/raw/McMaster Database with Numerical QuAL Scores.xlsx")


    


# +

for index, row in masterdb.iterrows():
#     print(index)
#     row['Survey N']
#     row['Question N']
    if row['dataSource'] == "Sas":
        
#         print(sasRow["Q1"])
    
        sasRow = RobMacSaskData[(RobMacSaskData['Survey N']==row['Survey N']) & (RobMacSaskData['Question N']==row['Question N'])]
        
        if len(sasRow)>1:
            print("More than 1 row with same survey and question number.")
            break
        
#         print(masterdb.loc[index])

        masterdb.loc[index,'RobMacCommentModified'] = sasRow["Feedback"].iat[0] #get the comment to cross check and this has slightly modified comments for typo etc.
    
        masterdb.loc[index,'RobMacQ1'] = sasRow["Q1"].iat[0]
        masterdb.loc[index,'RobMacQ2'] = sasRow["Q2"].iat[0]
        masterdb.loc[index,'RobMacQ3'] = sasRow["Q3"].iat[0]
    
    if row['dataSource'] == "Mac":   
        macRow = RobMacMacData[(RobMacMacData['Survey N']==row['Survey N']) & (RobMacMacData['Question N']==row['Question N'])]
        
        if len(macRow)>1:
            print("More than 1 row with same survey and question number on Mac Data.")
            break

        masterdb.loc[index,'RobMacCommentModified'] = macRow["Please provide comments about how the resident performed on THIS specific EPA."].iat[0] #get the comment to cross check
    
        masterdb.loc[index,'RobMacQ1'] = macRow["Q1"].iat[0]
        masterdb.loc[index,'RobMacQ2'] = macRow["Q2"].iat[0]
        masterdb.loc[index,'RobMacQ3'] = macRow["Q3"].iat[0]            
    
#     break

masterdb['RobMacQualScore'] = masterdb['RobMacQ1'] + masterdb['RobMacQ2'] + masterdb['RobMacQ3']

masterdb.to_excel("masterdb.xlsx")
# -

masterdb.columns
