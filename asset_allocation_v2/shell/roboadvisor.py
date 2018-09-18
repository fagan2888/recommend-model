# -*- coding: utf-8 -*-
__author__ = "Likun Liu"

import logging
import logging.config
import json
import os
import sys
import click
import time
import itertools
import pandas as pd
import Const
from Const import datapath

# import CommandModelRisk
import CommandExport
import CommandFund
import CommandPool
import CommandPortfolio
import CommandCompositeAsset
import CommandMarkowitz
import CommandRiskManage
import CommandTiming
import CommandHighlow
import CommandOnline
import CommandInvestor
import CommandExchangeRateIndex
import CommandUtil
import CommandAnalysis
import CommandImport
import CommandStockFactor
import CommandMacroTiming
import CommandFactorCluster
import CommandView
import CommandFundFactor
import CommandFundCluster
import CommandFundDecomp
import CommandIndexCluster
import CommandIndexFactor

from util import ProgressBar
from util.xdebug import dd
from db import *


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
            list(range(0, 10)), label='update nav',
            item_show_func=lambda x: "subtask %d" % x if x is not None else '', fill_char='#', empty_char='-',
            bar_template='%(label)-25s  [%(bar)s]  %(info)s', width=0) as bar:
        for i in bar:
            time.sleep(0.3)

@roboadvisor.command()
@click.option('--pool/--no-pool', 'optpool', default=True, help='include pool command group with in batch')
@click.option('--timing/--no-timing', 'opttiming', default=True, help='include timing command group with in batch')
@click.option('--reshape/--no-reshape', 'optreshape', default=False, help='include reshape command group with in batch')
@click.option('--riskmgr/--no-riskmgr', 'optriskmgr', default=True, help='include riskmgr command group with in batch')
@click.option('--markowitz/--no-markowitz', 'optmarkowtiz', default=True, help='include markowitz command group with in batch')
@click.option('--highlow/--no-highlow', 'opthighlow', default=True, help='include highlow command group with in batch')
@click.option('--portfolio/--no-portfolio', 'optportfolio', default=True, help='include portfolio command group with in batch')
@click.option('--start-date', 'startdate', default='2012-07-27', help='start date to calc')
@click.option('--end-date', 'enddate', help='end date to calc')
@click.option('--turnover-markowitz', 'optturnoverm', type=float, default=0, help='fitler portfolio by turnover')
@click.option('--turnover-portfolio', 'optturnoverp', type=float, default=0.4, help='fitler portfolio by turnover')
@click.option('--bootstrap/--no-bootstrap', 'optbootstrap', default=True, help='use bootstrap or not')
@click.option('--bootstrap-count', 'optbootcount', type=int, default=0, help='use bootstrap or not')
@click.option('--cpu-count', 'optcpu', type=int, default=0, help='how many cpu to use, (0 for all available)')
@click.option('--wavelet/--no-wavelet', 'optwavelet', default=False, help='use wavelet filter or not')
@click.option('--wavelet-filter-num', 'optwaveletfilternum', default=2, help='use wavelet filter num')
@click.option('--high', 'opthigh', type=int, help='specify markowitz high id when --no-markowitz')
@click.option('--low', 'optlow', type=int, help='specify markowitz low id when --no-markowitz specified')
@click.option('--ratio', 'optratio', type=int, help='specify highlow when --no-highlow specified ')
@click.option('--online/--no-online', 'optonline', default=False, help='include online instance for timing and riskmgr')
@click.option('--replace/--no-replace', 'optreplace', default=False, help='replace existed instance')
@click.option('--riskctrl/--no-riskctrl', 'optriskctrl', default=True, help='no riskmgr for highlow')
@click.option('--new/--no-new', 'optnew', default=False, help='use new framework')
@click.option('--append/--no-append', 'optappend', default=False, help='append pos or not')
@click.option('--id', 'optid', help='specify which id to run (only works for new framework)')
@click.pass_context
def run(ctx, optid, optpool, opttiming, optreshape, optriskmgr, optmarkowtiz, opthighlow, optportfolio, startdate, enddate, optturnoverm, optturnoverp, optbootstrap, optbootcount, optcpu, optwavelet, optwaveletfilternum, opthigh, optlow, optratio, optonline, optreplace, optriskctrl, optnew, optappend):
    '''run all command in batch
    '''
    if optpool:
        ctx.invoke(CommandPool.nav)
        #ctx.invoke(CommandPool.nav, optid='92401')

    if opttiming:
        ctx.invoke(CommandTiming.timing, optonline=optonline)

    if optreshape:
        pass

    if optriskmgr:
        ctx.invoke(CommandRiskManage.riskmgr, optonline=optonline)


    if optnew:
        run_new(ctx, optid=optid, optmarkowtiz=optmarkowtiz, opthighlow=opthighlow, optportfolio=optportfolio, optwavelet=optwavelet, optappend=optappend)

    else:
        if optmarkowtiz:
            ctx.invoke(CommandMarkowitz.markowitz, short_cut='high', startdate=startdate, optturnover=optturnoverm, optbootstrap=optbootstrap, optbootcount=optbootcount, optcpu=optcpu,optwavelet=optwavelet, optwaveletfilternum=optwaveletfilternum, optreplace=optreplace)
            ctx.invoke(CommandMarkowitz.markowitz, short_cut='low', startdate=startdate, optturnover=optturnoverm, optbootstrap=False, optbootcount=optbootcount, optcpu=optcpu)
        else:
            if opthigh is None:
                click.echo(click.style("--high required when --no-markowitz specified!", fg="red"))
                return
            else:
                ctx.obj['markowitz.high'] = opthigh
            if optlow is None:
                click.echo(click.style("--low required when --no-markowitz specified!", fg="red"))
            else:
                ctx.obj['markowitz.low'] = optlow

        if opthighlow:
            ctx.invoke(CommandHighlow.highlow, optreplace=optreplace, optriskmgr=optriskctrl)
        else:
            if optratio is None:
                click.echo(click.style("--ratio required when --no-highlow specified!", fg="red"))
            else:
                ctx.obj['highlow'] = optratio

        if optportfolio:
            ctx.invoke(CommandPortfolio.portfolio, optreplace=optreplace, optturnover=optturnoverp, optenddate=enddate)

        if optwavelet:
            ctx.invoke(CommandWavelet.filtering)

def run_new(ctx, optid, optmarkowtiz, opthighlow, optportfolio, optwavelet, optappend):
    '''
    new framework batchly
    '''

    ht = {}
    if optid is not None:
        gids = [s.strip() for s in optid.split(',')]

        df_portfolio = asset_ra_portfolio.load(gids)

        gids = gids + df_portfolio['ra_ratio_id'].tolist()

        df_highlow = asset_mz_highlow.load(gids)

        gids = gids + df_highlow['mz_markowitz_id'].tolist() + df_highlow['mz_high_id'].tolist() + df_highlow['mz_low_id'].tolist()

        df_markowitz = asset_mz_markowitz.load(gids)

        gids = gids + df_markowitz['globalid'].tolist()

        gids = [x for x in list(set(gids)) if x is not None and x != ""]

        gids = sorted(gids)

        ht = {k:list(v) for k,v in itertools.groupby(sorted(gids), key=lambda x: x[0:x.find('.')])}

    #sys.exit(0)
    if optmarkowtiz:
        if ht.get('MZ') is None:
            ctx.invoke(CommandMarkowitz.markowitz, optnew=True, optappend=optappend)
        else:
            tmpid =','.join(ht.get('MZ'))
            ctx.invoke(CommandMarkowitz.markowitz, optnew=True, optid=tmpid, optappend=optappend)

    if opthighlow:
        if ht.get('HL') is None:
            ctx.invoke(CommandHighlow.highlow, optnew=True)
        else:
            tmpid =','.join(ht.get('HL'))
            ctx.invoke(CommandHighlow.highlow, optnew=True, optid=tmpid)

    if optportfolio:
        if ht.get('PO') is None:
            ctx.invoke(CommandPortfolio.portfolio, optnew=True)
        else:
            tmpid =','.join(ht.get('PO'))
            ctx.invoke(CommandPortfolio.portfolio, optnew=True, optid=tmpid)

    if optwavelet:
        ctx.invoke(CommandWavelet.filtering, optnew=True)

@roboadvisor.command()
@click.option('--from', 'optfrom', default=True, help='--from id')
@click.option('--to', 'optto', default=True, help='--to id')
@click.option('--name', 'optname', default=True, help='name')
#@click.option('--riskmgr/--no-riskmgr', 'optriskmgr', default=True, help=u'include riskmgr command group with in batch')
#@click.option('--markowitz/--no-markowitz', 'optmarkowtiz', default=True, help=u'include markowitz command group with in batch')
@click.pass_context
def cp(ctx, optfrom, optto, optname):

    pass



if __name__=='__main__':
    # model.add_command(CommandModelRisk.risk)
    # pool.add_command(CommandPool.stock)
    # pool.add_command(CommandPool.bond)
    roboadvisor.add_command(CommandPortfolio.portfolio)
    roboadvisor.add_command(CommandExport.export)
    roboadvisor.add_command(CommandFund.fund)
    roboadvisor.add_command(CommandPool.pool)
    roboadvisor.add_command(CommandCompositeAsset.composite)
    roboadvisor.add_command(CommandMarkowitz.markowitz)
    roboadvisor.add_command(CommandRiskManage.riskmgr)
    roboadvisor.add_command(CommandTiming.timing)
    roboadvisor.add_command(CommandHighlow.highlow)
    roboadvisor.add_command(CommandOnline.online)
    roboadvisor.add_command(CommandInvestor.investor)
    roboadvisor.add_command(CommandExchangeRateIndex.exrindex)
    roboadvisor.add_command(CommandUtil.util)
    roboadvisor.add_command(CommandAnalysis.analysis)
    roboadvisor.add_command(CommandImport.import_command)
    roboadvisor.add_command(CommandStockFactor.sf)
    roboadvisor.add_command(CommandMacroTiming.mt)
    roboadvisor.add_command(CommandFactorCluster.fc)
    roboadvisor.add_command(CommandView.view)
    roboadvisor.add_command(CommandFundFactor.ff)
    roboadvisor.add_command(CommandFundCluster.fuc)
    roboadvisor.add_command(CommandFundDecomp.fd)
    roboadvisor.add_command(CommandIndexFactor.indexfactor)
    roboadvisor.add_command(CommandIndexCluster.ic)
    roboadvisor(obj={})
