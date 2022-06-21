import pandas as pd
import numpy as np
import typer 
from pathlib import Path
import logging as log

def _open_and_merge(mac_path : Path, sas_path : Path, qual_dir : Path):
    sasdf = pd.read_excel(sas_path)

    SasCommentCol = "Feedback"
    SasEPACol = "EPA"
    SasRatingCol = 'Rating'
    SasRawNumber = 'Random Number'
    SasObserverType = 'Observer Type'
    SasObserverSpecialty = 'EM/PEM vs off-service'

    macdf = pd.read_excel(mac_path)

    MacCommentCol = "Please provide comments about how the resident performed on THIS specific EPA."
    MacEPACol = "EPA"
    MacRatingCol = "Based on my observation of the resident's performance on this EPA today:"
    MacRawNumber = 'Number'
    MacGenderRes = 'GenderRes'
    MacGenderFac = 'GenderFac'
    MacObserverType = 'Unnamed: 6'
    MacObserverSpecialty = 'Type'
    MacPGY = 'PGY'

    q1 = "Does the rater provide sufficient evidence about performance?"
    q2 = "Does the rater provide a suggestion for improvement?"
    q3 = "Is the rater's suggestion linked to the behaviour described?"

    isResidentQuestion = "Are you a resident or an attending physician?"

    currentSurveyNo = 0
    currentQuestionNo = 0
    # masterdb = pd.DataFrame(columns = ["commentId", "dataSource", "NumberFromRawData", "comment", "rating", "Survey N", "Question N", "q1p1", "q1p2", "q2p1", "q2p2", "q3p1", "q3p2"])

    ### process sask data ###
    masterdb = []
    for index, row in sasdf.iterrows():
        if currentSurveyNo != row['Survey N']:
            currentSurveyNo = row['Survey N']
            if currentSurveyNo < 10:
                surveyNo = "0" + str(currentSurveyNo)
            else:
                surveyNo = str(currentSurveyNo)
            currentData = pd.read_excel(qual_dir / ("EPA Narrative Comment Survey v"+ surveyNo + ".xlsx")).drop(0).sort_values(isResidentQuestion) #Sor the residency question so that participant 1 in the masterdb answers as resident and the participant 2 is attending.

        currentQuestionNo = row['Question N']
        if currentQuestionNo == 1:
            questionNo = ""
        else:
            questionNo = "." + str(row['Question N'] - 1) #df column name determination
        
        q1Col = q1 + questionNo        
        q2Col = q2 + questionNo
        q3Col = q3 + questionNo

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
       
        masterdb.append(newrow)
        
    ### process mcmaster data ###
    for index, row in macdf.iterrows():
        if currentSurveyNo != row['Survey N']:
            currentSurveyNo = row['Survey N']
            if currentSurveyNo < 10:
                surveyNo = "0" + str(currentSurveyNo)
            else:
                surveyNo = str(currentSurveyNo)
            currentData = pd.read_excel(qual_dir / ("EPA Narrative Comment Survey v"+ surveyNo + ".xlsx")).drop(0)

        currentQuestionNo = row['Question N']
        if currentQuestionNo == 1:
            questionNo = ""
        else:
            questionNo = "." + str(row['Question N'] - 1) #df column name determination
        
        q1Col = q1 + questionNo        
        q2Col = q2 + questionNo
        q3Col = q3 + questionNo
        
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
        
        masterdb.append(newrow) # = pd.concat([masterdb, pd.Series(newrow)], ignore_index=True)

    masterdb = pd.DataFrame(masterdb)
    return masterdb

def _process_qual_score(masterdb : pd.DataFrame):
    ### Transform data to fit the QUAL score metrics ###
    masterdb['q1p1T'] = masterdb['q1p1'].replace([1,2,3,4],[3,2,1,0])
    masterdb['q1p2T'] = masterdb['q1p2'].replace([1,2,3,4],[3,2,1,0])

    masterdb['q2p1T'] = masterdb['q2p1'].replace([1,2],[1,0])
    masterdb['q2p2T'] = masterdb['q2p2'].replace([1,2],[1,0])

    masterdb['q3p1T'] = masterdb['q3p1'].replace([1,2],[1,0])
    masterdb['q3p2T'] = masterdb['q3p2'].replace([1,2],[1,0])

    #### Calculate qual scores for each participant ###
    masterdb['P1QualScore'] = masterdb['q1p1T'] + masterdb['q2p1T'] + masterdb['q3p1T']
    masterdb['P2QualScore'] = masterdb['q1p2T'] + masterdb['q2p2T'] + masterdb['q3p2T']

    ### Compare participant answer to find perfect match and descrepencies  ###
    masterdb['Q1Match'] = np.where(masterdb['q1p1T'] == masterdb['q1p2T'],"TRUE","FALSE")
    masterdb['Q2Match'] = np.where(masterdb['q2p1T'] == masterdb['q2p2T'],"TRUE","FALSE")
    masterdb['Q3Match'] = np.where(masterdb['q3p1T'] == masterdb['q3p2T'],"TRUE","FALSE")
    masterdb['perfectMatch'] = np.where((masterdb['Q1Match'] == "TRUE") & (masterdb['Q2Match'] == "TRUE") & (masterdb['Q3Match'] == "TRUE"),"TRUE","FALSE")

    return masterdb

def _merge_rob_mac_scoring(masterdb : pd.DataFrame, rob_mac_sas_path: Path, rob_mac_mac_path: Path):
    ### Merge Rob and Mac's scorrings on previous data ###
    RobMacSaskData = pd.read_excel(rob_mac_sas_path)
    RobMacMacData = pd.read_excel(rob_mac_mac_path)

    for index, row in masterdb.iterrows():
        if row['dataSource'] == "Sas":
            sasRow = RobMacSaskData[(RobMacSaskData['Survey N']==row['Survey N']) & (RobMacSaskData['Question N']==row['Question N'])]
            if len(sasRow)>1:
                print("More than 1 row with same survey and question number.")
                break

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

    masterdb['RobMacQualScore'] = masterdb['RobMacQ1'] + masterdb['RobMacQ2'] + masterdb['RobMacQ3']
    return masterdb

def _impute_macrob_score_for_imperfect_matches(df: pd.DataFrame):
    dPerfect = df[df['perfectMatch'] == "TRUE"]
    dNonPerfect = df[df['perfectMatch'] == "FALSE"]

    df['Q1'] = dNonPerfect['RobMacQ1']
    df['Q2'] = dNonPerfect['RobMacQ2']
    df['Q3'] = dNonPerfect['RobMacQ3']
    df['QUAL'] = dNonPerfect['RobMacQualScore']

    # now fill the blanks from the original ratings.
    # it does not matter getting P1 or P2 score as they are perfect match
    df['Q1'].fillna(dPerfect['q1p1T'], inplace=True)
    df['Q2'].fillna(dPerfect['q2p1T'], inplace=True)
    df['Q3'].fillna(dPerfect['q3p1T'], inplace=True)
    df['QUAL'].fillna(dPerfect['P1QualScore'], inplace=True)

    #calculate sum of qual scores and compare with the previous manually summed values to determine if they check out.
    df['summedQs'] = df['Q1']+df['Q2']+df['Q3']
    comparison_QUALScore_columns = np.where(df['summedQs'] == df['QUAL'], True, False)
    df["isQUALequal"] = comparison_QUALScore_columns
    log.info(f'Number of manually summed columns that equal auto-summed QuAL scores:\n{df["isQUALequal"].value_counts()}')
    log.info('Unequal will be replaced by auto-summed QuAL scores.')
    # for those that don't replace with the calculated qual scores
    df.loc[(df.isQUALequal == False),'QUAL'] = df['summedQs']
    # get rid of helper columns
    df.drop(['summedQs', 'isQUALequal'], axis=1, inplace=True)

    return df 

def main(mac_path: Path = typer.Option('data/raw/comments/mcmaster-database-de-identified-comments.xlsx', exists=True, dir_okay=False), 
         sas_path: Path = typer.Option('data/raw/comments/sask-database-de-identified-comments.xlsx', exists=True, dir_okay=False),
         qual_dir: Path = typer.Option('data/raw/qual-ratings/', exists=True, dir_okay=True, file_okay=False),
         rob_mac_mac_path: Path = typer.Option('data/raw/rob-mac-qual-ratings/mcmaster-database-with-numerical-qual-scores.xlsx', exists=True, dir_okay=False),
         rob_mac_sas_path: Path = typer.Option('data/raw/rob-mac-qual-ratings/sask-database-with-numerical-qual-scores.xlsx', exists=True, dir_okay=False),
         output_path: Path = typer.Option('data/processed/masterdb.xlsx'),
         log_level: str = typer.Option('INFO')):

    params = locals()
    log.basicConfig(level=log_level)
    log.info("Generating final dataset from raw data...")
    log.debug(f"Parameters:\n{params}")
        
    # open the Mac/Sask raw data and merge them into one file
    log.info('Merging Mac and Sask data...')
    masterdb = _open_and_merge(mac_path, sas_path, qual_dir)

    # calculate the QuAL scores from the responeses in the merged data
    log.info('Calculating QuAL scores...')
    masterdb = _process_qual_score(masterdb)

    # add the corrected QuAL scores from Rob & Mac
    log.info('Correcting QuAL scores with Rob/Mac hand-coding...')
    masterdb = _merge_rob_mac_scoring(masterdb, rob_mac_sas_path, rob_mac_mac_path)

    # impute the corrected QuAL scores from above if they don't match
    log.info('Imputing corrected scores where necessary...')
    masterdb = _impute_macrob_score_for_imperfect_matches(masterdb)

    # output the result
    log.info(f'Saving to {output_path}')
    masterdb.to_excel(output_path)

if __name__ == '__main__':
    typer.run(main)
    exit(0)
