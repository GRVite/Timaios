#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:08:43 2020

@author: vite
"""

import pandas as pd
from vitemod import *



#Access google sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
##enter the id of your google sheet
SAMPLE_SPREADSHEET_ID_input = '1DiJMx6G9IhU_X_QY6NTWbqBWh5CvvLsoQVdo4IN0KXc'
SAMPLE_RANGE_NAME = 'A1:AA100'
values = accessgoogle(SAMPLE_SPREADSHEET_ID_input, SAMPLE_RANGE_NAME)
df=pd.DataFrame(values[1:], columns=values[0])


#rootDir = '/media/3TBHDD/Data'
rootDir = '/Users/vite/navigation_system/Data'
ID = 'A4405'
session = 'A4405-200312'
episodes = df[df['Session']=='A4405-200312']["Episodes"].values.tolist()


import pexpect
child = pexpect.spawn('python main_opto.py')
child.expect("ID")
child.sendline(ID)
child.expect("Session")
child.sendline(session)
child.expect("Root directory")
child.sendline(rootDir)
child.expect(pexpect.EOF) 


child = pexpect.spawn('python main_opto.py')
child.expect("ID")
child.send(ID)
child.expect("Session")
child.send(session)
child.expect("Root directory")
child.send(rootDir)
child.expect(pexpect.EOF) 