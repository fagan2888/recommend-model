#coding=utf-8
'''
Created on: Mar. 19, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
import click
from sqlalchemy import MetaData, Table, select, func
import numpy as np
import pandas as pd
from copy import deepcopy
from tabulate import tabulate
# from ipdb import set_trace
sys.path.append('shell')
from db import database
from db import factor_sp_stock_portfolio, factor_sp_stock_portfolio_argv, factor_sp_stock_portfolio_pos, factor_sp_stock_portfolio_nav
from trade_date import ATradeDate
import stock_portfolio

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--list/--no-list', 'optlist', default=False, help='print stock paortfolio list')
@click.option('--id', 'optid', help='stock portfolio id')
@click.option('--begin-date', 'begin_date', default='2006-01-01', help='begin date to calculate')
@click.option('--end-date', 'end_date', help='end date to calculate')
@click.option('--cpu-count', 'cpu_count', type=int, default=0, help='how many cpu to use, (0 for all available)')
@click.pass_context
def sp(ctx, optlist, optid, begin_date, end_date, cpu_count):
    ''' stock portfolio
    '''

    if ctx.invoked_subcommand is None:

        if optlist:

            df_stock_portfolio_info = factor_sp_stock_portfolio.load()
            print(tabulate(df_stock_portfolio_info, headers='keys', tablefmt='psql'))

            return

        else:

            ctx.invoke(pos_n_nav, optid=optid, begin_date=begin_date, end_date=end_date, cpu_count=cpu_count)


@sp.command()
@click.option('--id', 'optid', help='stock portfolio id to update')
@click.option('--type', 'opttype', default='0,1', help='which type to run')
@click.option('--begin-date', 'begin_date', default='2006-01-01', help='begin date to calculate')
@click.option('--end-date', 'end_date', help='end date to calculate')
@click.option('--cpu-count', 'cpu_count', type=int, default=0, help='how many cpu to use, (0 for all available)')
@click.pass_context
def pos_n_nav(ctx, optid, opttype, begin_date, end_date, cpu_count):
    ''' calculate pos and nav of stock portfolio to update
    '''

    if optid is not None:
        list_portfolio_id = [s.strip() for s in optid.split(',')]
    else:
        list_portfolio_id = None

    list_type = [s.strip() for s in opttype.split(',')]

    if list_portfolio_id is not None:
        df_stock_portfolio_info = factor_sp_stock_portfolio.load(portfolio_ids=list_portfolio_id)
    else:
        df_stock_portfolio_info = factor_sp_stock_portfolio.load(types=list_type)

    for _, stock_portfolio_info in df_stock_portfolio_info.iterrows():
        pos_n_nav_update(stock_portfolio_info, begin_date, end_date)

def pos_n_nav_update(stock_portfolio_info, begin_date, end_date):

    stock_portfolio_id = stock_portfolio_info.name
    algo = stock_portfolio_info.loc['sp_algo']

    df_argv = factor_sp_stock_portfolio_argv.load(portfolio_id=stock_portfolio_id)
    kwargs = df_argv.loc[stock_portfolio_id].sp_value.to_dict()

    list_int_arg = [
        'look_back',
        'exclusion',
    ]

    list_float_arg = [
        'percentage'
    ]

    for arg in list_int_arg:
        if kwargs.get(arg) is not None:
            kwargs[arg] = int(kwargs.get(arg))

    for arg in list_float_arg:
        if kwargs.get(arg) is not None:
            kwargs[arg] =float(kwargs.get(arg))

    period = kwargs.get('period', 'day')

    if period == 'day':
        kwargs['reindex'] = ATradeDate.trade_date(begin_date=begin_date, end_date=end_date).rename('trade_date')
    elif period == 'week':
        kwargs['reindex'] = ATradeDate.week_trade_date(begin_date=begin_date, end_date=end_date).rename('trade_date')
    elif period == 'month':
        kwargs['reindex'] = ATradeDate.month_trade_date(begin_date=begin_date, end_date=end_date).rename('trade_date')

    else:
        click.echo(click.style(f'\n Period {period} is unknown for stock portfolio {stock_portfolio_id}.', fg='red'))
        return

    try:

        class_name = f'StockPortfolio{algo}'
        cls = getattr(stock_portfolio, class_name)

        class_stock_portfolio = cls(**kwargs)
        click.echo(click.style(f'\n Stock data for stock portfolio {stock_portfolio_id} loaded.', fg='yellow'))

        class_stock_portfolio.calc_portfolio_nav()
        click.echo(click.style(f'\n Nav of stock portfolio {stock_portfolio_id} calculated.', fg='yellow'))

        df_pos = deepcopy(class_stock_portfolio.df_stock_pos_adjusted)
        df_nav = pd.DataFrame({'nav': class_stock_portfolio.ser_portfolio_nav, 'inc': class_stock_portfolio.ser_portfolio_inc})

    except AttributeError:

        click.echo(click.style(f'\n Algo {algo} is unknown for stock portfolio {stock_portfolio_id}.', fg='red'))
        return

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    table_sp_pos = Table('sp_stock_portfolio_pos', metadata, autoload=True)
    table_sp_nav = Table('sp_stock_portfolio_nav', metadata, autoload=True)

    table_sp_pos.delete(table_sp_pos.c.globalid==stock_portfolio_id).execute()
    table_sp_nav.delete(table_sp_nav.c.globalid==stock_portfolio_id).execute()

    df_pos = df_pos.stack().rename('sp_sk_pos').reset_index().rename(columns={'trade_date': 'sp_date', 'stock_id': 'sp_sk_id'})
    df_pos['globalid'] = stock_portfolio_id
    df_pos = df_pos.loc[df_pos.sp_sk_pos>0.0].set_index(['globalid', 'sp_date', 'sp_sk_id'])

    df_nav = df_nav.reset_index().rename(columns={'trade_date': 'sp_date', 'nav': 'sp_nav', 'inc': 'sp_inc'})
    df_nav['globalid'] = stock_portfolio_id
    df_nav = df_nav.set_index(['globalid', 'sp_date'])

    factor_sp_stock_portfolio_pos.save(stock_portfolio_id, df_pos)
    factor_sp_stock_portfolio_nav.save(stock_portfolio_id, df_nav)

    click.echo(click.style(f'\n Successfully updated pos and nav of stock portfolio {stock_portfolio_info.name}!', fg='green'))


@sp.command()
@click.option('--src', 'optsrc', help='src id of stock portfolio to copy from')
@click.option('--dst', 'optdst', help='dst id of stock portfolio to copy to')
@click.pass_context
def copy(ctx, optsrc, optdst):
    ''' create new stock portfolio by copying existed one
    '''

    if optsrc is None or optdst is None:

        click.echo(click.style(f'\n Both src id and dst id are required to perform copy.', fg='red'))
        return

    src_portfolio_id = optsrc
    dst_portfolio_id = optdst

    # copy sp_stock_portfolio
    df_stock_portfolio_info = factor_sp_stock_portfolio.load(portfolio_ids=[src_portfolio_id])
    df_stock_portfolio_info.rename(index={src_portfolio_id: dst_portfolio_id}, inplace=True)
    factor_sp_stock_portfolio.save(dst_portfolio_id, df_stock_portfolio_info)

    # copy sp_stock_portfolio_argv
    df_stock_portfolio_argv = factor_sp_stock_portfolio_argv.load(portfolio_id=src_portfolio_id)
    df_stock_portfolio_argv.rename(index={src_portfolio_id: dst_portfolio_id}, level='globalid', inplace=True)
    factor_sp_stock_portfolio_argv.save(dst_portfolio_id, df_stock_portfolio_argv)

    click.echo(click.style(f'\n Successfully created stock portfolio {dst_portfolio_id}!', fg='green'))

