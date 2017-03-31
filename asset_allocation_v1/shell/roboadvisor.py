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
import CommandExport
import CommandFund
import CommandPool
import CommandPortfolio
import CommandCompositeAsset
import CommandMarkowitz
import CommandReshape
import CommandRiskManage
import CommandTiming
import CommandHighlow

from util import ProgressBar


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
def model(ctx):
    '''run models
    '''
    click.echo("model")

@roboadvisor.group()  
@click.pass_context
def nav(ctx):
    '''fund pool group
    '''
    pass

@roboadvisor.command()  
@click.pass_context
def test(ctx):
    '''test code
    '''
    # pbar = click.progressbar(length=10, label='update nav')
    # iterable=iterable, length=length, show_eta=show_eta,
    #                    show_percent=show_percent, show_pos=show_pos,
    #                    item_show_func=item_show_func, fill_char=fill_char,
    #                    empty_char=empty_char, bar_template=bar_template,
    #                    info_sep=info_sep, file=file, label=label,
    #                    width=width, color=color)

    with ProgressBar.ProgressBar(
            range(0, 10), label='update nav',
            item_show_func=lambda x: "subtask %d" % x if x is not None else '', fill_char='#', empty_char='-',
            bar_template='%(label)-25s  [%(bar)s]  %(info)s', width=0) as bar:
        for i in bar:
            time.sleep(0.3)

@roboadvisor.command()
@click.option('--pool/--no-pool', 'optpool', default=True, help=u'include pool command group with in batch')
@click.option('--timing/--no-timing', 'opttiming', default=True, help=u'include timing command group with in batch')
@click.option('--reshape/--no-reshape', 'optreshape', default=True, help=u'include reshape command group with in batch')
@click.option('--riskmgr/--no-riskmgr', 'optriskmgr', default=True, help=u'include riskmgr command group with in batch')
@click.option('--markowitz/--no-markowitz', 'optmarkowtiz', default=True, help=u'include markowitz command group with in batch')
@click.option('--highlow/--no-highlow', 'opthighlow', default=True, help=u'include highlow command group with in batch')
@click.option('--portfolio/--no-portfolio', 'optportfolio', default=True, help=u'include portfolio command group with in batch')
@click.option('--start-date', 'startdate', default='2012-07-27', help=u'start date to calc')
@click.option('--turnover', 'optturnover', type=float, default=0, help=u'fitler by turnover')
@click.option('--bootstrap/--no-bootstrap', 'optbootstrap', default=True, help=u'use bootstrap or not')
@click.option('--bootstrap-count', 'optbootcount', type=int, default=0, help=u'use bootstrap or not')
@click.option('--cpu-count', 'optcpu', type=int, default=0, help=u'how many cpu to use, (0 for all available)')
@click.option('--online/--no-online', 'optonline', default=False, help=u'include online instance for timing and riskmgr')
@click.pass_context
def run(ctx, optpool, opttiming, optreshape, optriskmgr, optmarkowtiz, opthighlow, optportfolio, startdate,  optturnover, optbootstrap, optbootcount, optcpu, optonline):
    '''run all command in batch
    '''
    if optpool:
        ctx.invoke(CommandPool.nav, optid='92101,92201,92301,92401')
        #ctx.invoke(CommandPool.nav, optid='92401')

    if opttiming:
        ctx.invoke(CommandTiming.timing, optonline=optonline)

    if optreshape:
        ctx.invoke(CommandReshape.reshape, optonline=optonline)

    if optriskmgr:
        ctx.invoke(CommandRiskManage.riskmgr, optonline=optonline)

    if optmarkowtiz:
        ctx.invoke(CommandMarkowitz.markowitz, short_cut='high', startdate=startdate, optturnover=optturnover, optbootstrap=optbootstrap, optbootcount=optbootcount, optcpu=optcpu)
        ctx.invoke(CommandMarkowitz.markowitz, short_cut='low', startdate=startdate, optturnover=optturnover, optbootstrap=optbootstrap, optbootcount=optbootcount, optcpu=optcpu)

    if opthighlow:
        ctx.invoke(CommandHighlow.highlow)

    if optportfolio:
        ctx.invoke(CommandPortfolio.portfolio)
    

if __name__=='__main__':
    model.add_command(CommandModelRisk.risk)
    nav.add_command(CommandNavStock.stock)
    # pool.add_command(CommandPool.stock)
    # pool.add_command(CommandPool.bond)
    roboadvisor.add_command(CommandPortfolio.portfolio)
    roboadvisor.add_command(CommandExport.export)
    roboadvisor.add_command(CommandFund.fund)
    roboadvisor.add_command(CommandPool.pool)
    roboadvisor.add_command(CommandCompositeAsset.composite)
    roboadvisor.add_command(CommandReshape.reshape)
    roboadvisor.add_command(CommandMarkowitz.markowitz)
    roboadvisor.add_command(CommandRiskManage.riskmgr)
    roboadvisor.add_command(CommandTiming.timing)
    roboadvisor.add_command(CommandHighlow.highlow)

    roboadvisor(obj={})  
    
