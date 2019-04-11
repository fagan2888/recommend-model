#coding=utf-8
'''
Created on: Mar. 19, 2019
Modified on: Apr. 8, 2019
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
import stock_portfolio
from db import database
from db import caihui_tq_sk_basicinfo
from db import factor_sp_stock_portfolio, factor_sp_stock_portfolio_argv, factor_sp_stock_portfolio_pos, factor_sp_stock_portfolio_nav
from trade_date import ATradeDate


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--list/--no-list', 'optlist', default=False, help='print list of stock paortfolios')
@click.option('--id', 'optid', help='stock portfolio ids')
@click.option('--type', 'opttype', help='stock portfolio types to update')
@click.option('--begin-date', 'begin_date', default='2006-01-01', help='begin date to calculate')
@click.option('--end-date', 'end_date', help='end date to calculate')
@click.option('--cpu-count', 'cpu_count', type=int, default=0, help='cpu count to use, (0 for all available)')
@click.pass_context
def sp(ctx, optlist, optid, opttype, begin_date, end_date, cpu_count):
    ''' stock portfolio
    '''

    if ctx.invoked_subcommand is None:

        if optlist:

            df_stock_portfolio_info = factor_sp_stock_portfolio.load_all()
            print(tabulate(df_stock_portfolio_info, headers='keys', tablefmt='psql'))

            return

        else:

            ctx.invoke(pos_n_nav, optid=optid, opttype=opttype, begin_date=begin_date, end_date=end_date, cpu_count=cpu_count)


@sp.command()
@click.option('--id', 'optid', help='stock portfolio ids to update')
@click.option('--type', 'opttype', help='stock portfolio types to update')
@click.option('--begin-date', 'begin_date', default='2006-01-01', help='begin date to calculate')
@click.option('--end-date', 'end_date', help='end date to calculate')
@click.option('--cpu-count', 'cpu_count', type=int, default=0, help='cpu count to use, (0 for all available)')
@click.pass_context
def pos_n_nav(ctx, optid, opttype, begin_date, end_date, cpu_count):
    ''' calculate pos and nav of stock portfolio to update
    '''

    if optid is not None:
        list_portfolio_id = [s.strip() for s in optid.split(',')]
    else:
        list_portfolio_id = None

    if opttype is not None:
        list_type = [s.strip() for s in opttype.split(',')]
    else:
        list_type = None

    if list_portfolio_id is None and list_type is None:

        click.echo(click.style(f'\n Either stock portfolio id or type is required to perform pos n nav.', fg='red'))
        return

    if list_portfolio_id is not None:
        df_stock_portfolio_info = factor_sp_stock_portfolio.load_by_id(portfolio_ids=list_portfolio_id)
    else:
        df_stock_portfolio_info = factor_sp_stock_portfolio.load_by_type(types=list_type)

    for _, stock_portfolio_info in df_stock_portfolio_info.iterrows():
        pos_n_nav_update(stock_portfolio_info, begin_date, end_date)

def pos_n_nav_update(stock_portfolio_info, begin_date, end_date):

    stock_portfolio_id = stock_portfolio_info.name
    stock_portfolio_type = stock_portfolio_info.loc['sp_type']
    algo = stock_portfolio_info.loc['sp_algo']

    df_argv = factor_sp_stock_portfolio_argv.load(portfolio_id=stock_portfolio_id)
    kwargs = df_argv.loc[stock_portfolio_id].sp_value.to_dict()

    list_int_arg = [
        'look_back',
        'exclusion'
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

    if kwargs['reindex'].size == 0:
        click.echo(click.style(f'\n Trade date index for stock portfolio {stock_portfolio_id} is empty.', fg='red'))
        return

    if stock_portfolio_id[:3] != 'CS.' and stock_portfolio_id[-2:] != '00':

        algo = f'Industry{algo}'
        kwargs['sw_industry_code'] = f'{stock_portfolio_id[-2:]}0000'

    try:

        class_name = f'StockPortfolio{algo}'
        cls = getattr(stock_portfolio, class_name)

    except AttributeError:

        click.echo(click.style(f'\n Algo {algo} is unknown for stock portfolio {stock_portfolio_id}.', fg='red'))
        return

    class_stock_portfolio = cls(**kwargs)
    click.echo(click.style(f'\n Stock data for stock portfolio {stock_portfolio_id} loaded.', fg='yellow'))

    if stock_portfolio_type == 0:
        class_stock_portfolio.calc_portfolio_nav(considering_status=False)
    elif stock_portfolio_type == 1:
        class_stock_portfolio.calc_portfolio_nav()
    else:
        click.echo(click.style(f'\n Type {stock_portfolio_type} is unknown for stock portfolio {stock_portfolio_id}.', fg='red'))

    df_pos = deepcopy(class_stock_portfolio.df_stock_pos_adjusted)
    df_nav = pd.DataFrame({'nav': class_stock_portfolio.ser_portfolio_nav, 'inc': class_stock_portfolio.ser_portfolio_inc})
    class_stock_portfolio.portfolio_analysis()
    click.echo(click.style(f'\n Nav of stock portfolio {stock_portfolio_id} calculated.', fg='yellow'))

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
@click.option('--id', 'optid', help='stock portfolio id')
@click.option('--name', 'optname', default='', help='stock portfolio name')
@click.option('--type', 'opttype', default='1', help='stock portfolio type')
@click.option('--algo', 'optalgo', default='MarketCap', help='stock portfolio algo')
@click.option('--argv', 'optargv', default='index_id:2070000191:指数ID,look_back:120:回测时长,period:day:调整间隔', help='stock portfolio argv')
@click.option('--override/--no-override', 'optoverride', default=False, help='override existing stock portfolio or not')
@click.pass_context
def create(ctx, optid, optname, opttype, optalgo, optargv, optoverride):
    ''' create new stock portfolio
    '''

    if optid is None:

        click.echo(click.style(f'\n Stock portfolio id is required to perform create.', fg='red'))
        return

    portfolio_id = optid
    portfolio_name = optname
    portfolio_type = opttype
    portfolio_algo = optalgo
    list_portfolio_argv = [[s2.strip() for s2 in s1.split(':')] for s1 in optargv.split(',')]

    if not optoverride and factor_sp_stock_portfolio.load_by_id(portfolio_ids=[portfolio_id]).shape[0] > 0:

        click.echo(click.style(f'\n Stock portfolio {portfolio_id} is existed.', fg='red'))
        return

    df_stock_portfolio_info = pd.DataFrame([[portfolio_name, portfolio_type, portfolio_algo]], columns=['sp_name', 'sp_type', 'sp_algo'])
    df_stock_portfolio_info['globalid'] = portfolio_id
    df_stock_portfolio_info = df_stock_portfolio_info.set_index('globalid')
    factor_sp_stock_portfolio.save(portfolio_id, df_stock_portfolio_info)

    df_stock_portfolio_argv = pd.DataFrame(list_portfolio_argv, columns=['sp_key', 'sp_value', 'sp_desc'])
    df_stock_portfolio_argv['globalid'] = portfolio_id
    df_stock_portfolio_argv = df_stock_portfolio_argv.set_index(['globalid', 'sp_key'])
    factor_sp_stock_portfolio_argv.save(portfolio_id, df_stock_portfolio_argv)

    click.echo(click.style(f'\n Successfully created stock portfolio {portfolio_id}!', fg='green'))


@sp.command()
@click.option('--src', 'optsrc', help='source id of stock portfolio to copy from')
@click.option('--dst', 'optdst', help='destination id of stock portfolio to copy to')
@click.option('--override/--no-override', 'optoverride', default=False, help='override existing stock portfolio or not')
@click.pass_context
def copy(ctx, optsrc, optdst, optoverride):
    ''' create new stock portfolio by copying existed one
    '''

    if optsrc is None or optdst is None:

        click.echo(click.style(f'\n Both source id and destination id are required to perform copy.', fg='red'))
        return

    src_portfolio_id = optsrc
    dst_portfolio_id = optdst

    if factor_sp_stock_portfolio.load_by_id(portfolio_ids=[src_portfolio_id]).shape[0] == 0:

        click.echo(click.style(f'\n Stock portfolio {src_portfolio_id} is not existed.', fg='red'))
        return

    if not optoverride and factor_sp_stock_portfolio.load_by_id(portfolio_ids=[dst_portfolio_id]).shape[0] > 0:

        click.echo(click.style(f'\n Stock portfolio {dst_portfolio_id} is existed.', fg='red'))
        return

    df_stock_portfolio_info = factor_sp_stock_portfolio.load_by_id(portfolio_ids=[src_portfolio_id])
    df_stock_portfolio_info.rename(index={src_portfolio_id: dst_portfolio_id}, inplace=True)
    factor_sp_stock_portfolio.save(dst_portfolio_id, df_stock_portfolio_info)

    df_stock_portfolio_argv = factor_sp_stock_portfolio_argv.load(portfolio_id=src_portfolio_id)
    df_stock_portfolio_argv.rename(index={src_portfolio_id: dst_portfolio_id}, level='globalid', inplace=True)
    factor_sp_stock_portfolio_argv.save(dst_portfolio_id, df_stock_portfolio_argv)

    click.echo(click.style(f'\n Successfully created stock portfolio {dst_portfolio_id}!', fg='green'))


@sp.command()
@click.option('--src', 'optsrc', help='source id of stock portfolio to devide from')
@click.option('--override/--no-override', 'optoverride', default=False, help='override existing stock portfolio or not')
@click.pass_context
def devide(ctx, optsrc, optoverride):
    ''' create new stock portfolio for industries
    '''

    if optsrc is None:

        click.echo(click.style(f'\n Src id is required to perform devide.', fg='red'))
        return

    src_portfolio_id = optsrc

    if src_portfolio_id[-2:] != '00':

        click.echo(click.style(f'\n Source id {src_portfolio_id} is invalid.', fg='red'))
        return

    if factor_sp_stock_portfolio.load_by_id(portfolio_ids=[src_portfolio_id]).shape[0] == 0:

        click.echo(click.style(f'\n Stock portfolio {src_portfolio_id} is not existed.', fg='red'))
        return

    if not optoverride and factor_sp_stock_portfolio.load_by_id(portfolio_ids=[src_portfolio_id[:-2]]).shape[0] > 1:

        click.echo(click.style(f'\n Stock portfolio for industries {src_portfolio_id[:-2]} is existed.', fg='red'))
        return

    sw_industry_code_pool = caihui_tq_sk_basicinfo.load_sw_industry_code_info()
    num_sw_industry = sw_industry_code_pool.shape[0]
    list_dst_portfolio_id = [f'{src_portfolio_id[:-2]}{sw_industry_code[:2]}' for sw_industry_code in sw_industry_code_pool.index]

    for dst_portfolio_id in list_dst_portfolio_id:

        df_stock_portfolio_info = factor_sp_stock_portfolio.load_by_id(portfolio_ids=[src_portfolio_id])
        df_stock_portfolio_info.rename(index={src_portfolio_id: dst_portfolio_id}, inplace=True)
        factor_sp_stock_portfolio.save(dst_portfolio_id, df_stock_portfolio_info)

        df_stock_portfolio_argv = factor_sp_stock_portfolio_argv.load(portfolio_id=src_portfolio_id)
        df_stock_portfolio_argv.rename(index={src_portfolio_id: dst_portfolio_id}, level='globalid', inplace=True)
        factor_sp_stock_portfolio_argv.save(dst_portfolio_id, df_stock_portfolio_argv)

    click.echo(click.style(f'\n Successfully created stock portfolios for industries {list_dst_portfolio_id}!', fg='green'))


@sp.command()
@click.option('--id', 'optid', help='stock portfolio id to remove')
@click.option('--except', 'optexcept', help='stock portfolio ids not to remove')
@click.pass_context
def remove(ctx, optid, optexcept):

    if optid is None:

        click.echo(click.style(f'\n Stock portfolio id is required to perform remove.', fg='red'))
        return

    if optexcept is not None:
        list_except_portfolio_id = [s.strip() for s in optexcept.split(',')]
    else:
        list_except_portfolio_id = []

    list_portfolio_id = list(factor_sp_stock_portfolio.load_by_id(portfolio_ids=[optid]).index)
    list_portfolio_id = list(set(list_portfolio_id)-set(list_except_portfolio_id))

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    table_sp = Table('sp_stock_portfolio', metadata, autoload=True)
    table_sp_argv = Table('sp_stock_portfolio_argv', metadata, autoload=True)
    table_sp_pos = Table('sp_stock_portfolio_pos', metadata, autoload=True)
    table_sp_nav = Table('sp_stock_portfolio_nav', metadata, autoload=True)

    for portfolio_id in list_portfolio_id:

        table_sp.delete(table_sp.c.globalid==portfolio_id).execute()
        table_sp_argv.delete(table_sp_argv.c.globalid==portfolio_id).execute()
        table_sp_pos.delete(table_sp_pos.c.globalid==portfolio_id).execute()
        table_sp_nav.delete(table_sp_nav.c.globalid==portfolio_id).execute()

    click.echo(click.style(f'\n Successfully remove stock portfolios {list_portfolio_id}!', fg='green'))

