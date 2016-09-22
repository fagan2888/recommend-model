# -*- coding: utf-8 -*-   
__author__ = "Likun Liu"  
  
import logging
import logging.config
import json
import os
import sys  
import click  
import time  

import Const
import GeneralizationPosition

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

    #config.load()        

@roboadvisor.group()  
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', type=click.File(mode='w'), default='-', help=u'file used to store final result')
@click.pass_context
def portfolio(ctx, datadir, output):
    ctx.obj['datadir'] = datadir
    ctx.obj['output'] = output
    Const.datadir = datadir
    pass;
    
@portfolio.command()  
# @click.option('-m', '--msg')  
# @click.option('--dry-run', is_flag=True, help=u'pretend to run')
# @click.option('--name', prompt='Your name', help='The person to greet.')
@click.pass_context
def simple(ctx):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    out = ctx.obj['output']
    #
    # 生成配置数据
    #
    all_code_position = GeneralizationPosition.risk_position()
    
    GeneralizationPosition.output_final_portfolio(all_code_position, out)

@portfolio.command()  
@click.pass_context
def optimize(ctx):
    '''generate final portfolio with optimized strategy (cost consider in).  
    '''
    out = ctx.obj['output']
    #
    # 生成配置数据
    #
    all_code_position = GeneralizationPosition.risk_position()
    
    GeneralizationPosition.output_portfolio(all_code_position, out)

@portfolio.command()  
@click.pass_context
def category(ctx):
    '''generate intemediate portfolio for different asset categories 
    '''
    out = ctx.obj['output']
    #
    # 生成配置数据
    #
    all_code_position = GeneralizationPosition.risk_position()
    
    GeneralizationPosition.output_category_portfolio(all_code_position, out)
    
@roboadvisor.group()  
@click.pass_context
def model(ctx):
    click.echo("model")
  
@model.command()
@click.pass_context
def risk(ctx):
    click.echo("const risk model, not integrated")

if __name__=='__main__':
    roboadvisor(obj={})  
