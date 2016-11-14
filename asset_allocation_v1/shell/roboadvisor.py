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
    setup_logging()
    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #config.load()        

@roboadvisor.group()  
@click.pass_context
def portfolio(ctx):
    ''' generate portfolios
    '''
    pass;
    
@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', type=click.File(mode='w'), default='-', help=u'file used to store final result')
# @click.option('-m', '--msg')  
# @click.option('--dry-run', is_flag=True, help=u'pretend to run')
# @click.option('--name', prompt='Your name', help='The person to greet.')
@click.pass_context
def simple(ctx, datadir, output):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    out = output
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    all_code_position = GeneralizationPosition.risk_position()
    
    GeneralizationPosition.output_final_portfolio(all_code_position, out)
    
@portfolio.command()  
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', type=click.File(mode='w'), default='-', help=u'file used to store final result')
@click.pass_context
def optimize(ctx, datadir, output):
    '''generate final portfolio with optimized strategy (cost consider in).  
    '''
    out = output
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    all_code_position = GeneralizationPosition.risk_position()
    
    GeneralizationPosition.output_portfolio(all_code_position, out)

@portfolio.command()  
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', type=click.File(mode='w'), default='-', help=u'file used to store final result')
@click.pass_context
def category(ctx, datadir, output):
    '''generate intemediate portfolio for different asset categories 
    '''
    out = output
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    all_code_position = GeneralizationPosition.risk_position()
    
    GeneralizationPosition.output_category_portfolio(all_code_position, out)

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
# @click.option('-m', '--msg')  
# @click.option('--dry-run', is_flag=True, help=u'pretend to run')
# @click.option('--name', prompt='Your name', help='The person to greet.')
@click.pass_context
def ncat(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_category()

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
# @click.option('-m', '--msg')  
# @click.option('--dry-run', is_flag=True, help=u'pretend to run')
# @click.option('--name', prompt='Your name', help='The person to greet.')
@click.pass_context
def nsimple(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_simple()

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.pass_context
def detail(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_detail()

@portfolio.command()  
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', default=None, help=u'file used to store final result')
@click.pass_context
def trade(ctx, datadir, output):
    '''generate final portfolio with optimized strategy (cost consider in).  
    '''
    Const.datadir = datadir
    if output is None:
        output = datapath('position-z.csv')
    with (open(output, 'w') if output != '-' else os.fdopen(os.dup(sys.stdout.fileno()), 'w')) as out:
        GeneralizationPosition.portfolio_trade(out)

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.pass_context
def stockavg(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_avg_simple()

    
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
    portfolio.add_command(CommandPortfolio.turnover)
    roboadvisor.add_command(CommandPool.pool)

    roboadvisor(obj={})  
    
