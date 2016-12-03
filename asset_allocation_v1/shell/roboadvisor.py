# -*- coding: utf-8 -*-   
__author__ = "Likun Liu"  
  
import logging
import logging.config
import json
import os
import sys  
import click  
import time  
import pandas as pd
import Const
from Const import datapath
import GeneralizationPosition

import CommandModelRisk
import CommandNavStock
import CommandPool
import CommandPortfolio
import CommandCompositeAsset
import CommandRiskManage


logger = logging.getLogger(__name__)

def setup_logging(
    default_path = './shell/logging.json', 
    default_level = logging.INFO,
    env_key = 'LOG_CFG'):
    
    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

        
@click.group(invoke_without_command=True)
@click.pass_context
def roboadvisor(ctx):
    default_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(default_dir, "logging.json")

    setup_logging(default_path=path)
    
    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #config.load()        
    
@roboadvisor.group()  
@click.pass_context
def measure(ctx):
    '''calc fund measure
    '''
    click.echo("model")
  
@measure.command()
@click.pass_context

def sharpe(ctx):
    '''calc fund sharpe ratio
    '''
    click.echo("Not integrated")

@measure.command()
@click.pass_context
def jensen(ctx):
    '''calc fund jensen alpha
    '''
    click.echo("Not integrated")

@measure.command()
@click.pass_context
def sortino(ctx):
    '''calc fund sortino ratio
    '''
    click.echo("Not integrated")

@measure.command()
@click.pass_context
def ppw(ctx):
    '''calc fund ppw measure
    '''
    click.echo("Not integrated")

@roboadvisor.group()  
@click.pass_context
def model(ctx):
    '''run models
    '''
    click.echo("model")

@roboadvisor.group()  
@click.pass_context
def nav(ctx):
    '''fund pool group
    '''
    click.echo("")

if __name__=='__main__':
    model.add_command(CommandModelRisk.risk)
    nav.add_command(CommandNavStock.stock)
    # pool.add_command(CommandPool.stock)
    # pool.add_command(CommandPool.bond)
    roboadvisor.add_command(CommandPortfolio.portfolio)
    roboadvisor.add_command(CommandPool.pool)
    roboadvisor.add_command(CommandCompositeAsset.composite)
    roboadvisor.add_command(CommandRiskManage.riskmgr)

    roboadvisor(obj={})  
    
