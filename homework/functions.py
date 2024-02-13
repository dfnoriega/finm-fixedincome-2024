from scipy import optimize as opt
import pandas as pd
import numpy as np

def continuous_from_discount(discount, maturity):
    '''Returns the continuous rate from the discount factor and the maturity'''
    return -np.log(discount)/maturity

def compounded_from_discount(discount, maturity, freq=2):
    '''Returns the compounded rate from the discount factor and the maturity'''
    return ((1/discount)**(1/(maturity*freq))-1)*freq

def discount_from_continuous(continuous, maturity):
    '''Returns the discount factor from the continuous rate and the maturity'''
    return np.exp(-continuous*maturity)

def discount_from_compounded(compounded, maturity, freq=2):
    '''Returns the discount factor from the compounded rate and the maturity'''
    return (1+compounded/freq)**(-maturity*freq)

def BootstrapData(df, maturities='maturity date', bid='bid', ask='ask'):
    '''Returns a dataframe with the minimum bid-ask spread for each bond maturity in the dataset.
    Specify maturity date column, bid column and ask column as strings.'''
    dates = df[maturities].unique()
    dates=pd.to_datetime(dates)
    df['bas']=(df[bid]+df[ask])
    tdf=pd.DataFrame(columns=df.columns)
    for i in dates:
        dfi = df[df[maturities]==i].copy()
        idx = dfi['bas'].idxmin()
        tdf.loc[len(tdf)] = dfi.loc[idx]
    tdf.drop(columns=['bas'], inplace=True)
    return tdf

def C_matrix(fdf, date, maturity_col, id_col, coupon_col, freq):
    '''Returns a dataframe with the cash flows of each bond in the dataset. Rounds to end of the month
    if less than 5 days away.
    date: string with to calculate cash flows from. Format: 'YYYY-MM-DD'
    maturity_col: string with the name of the column with the maturity date
    id_col: string with the name of the column with the bond id
    coupon_col: string with the name of the column with the coupon rate
    freq: integer with the number of coupon payments per year
    fdf: dataframe with the bond data'''
    tCdf = []
    for i in range(len(fdf)):
        row_data = {}
        crspid = fdf.iloc[i][id_col]
        maturity = pd.to_datetime(fdf.iloc[i][maturity_col]).strftime('%Y-%m-%d')
        coupon = fdf.iloc[i][coupon_col]
        row_data[id_col] = crspid
        row_data[maturity] = 100 + (coupon / freq)
        maturity = pd.to_datetime(maturity) - pd.DateOffset(months=int(12/freq))
        while maturity > pd.to_datetime(date):
            tmat = maturity
            if pd.to_datetime(maturity).day > pd.to_datetime(maturity+pd.offsets.MonthEnd(0)).day-5:
                tmat = tmat + pd.offsets.MonthEnd(0)
            row_data[tmat.strftime('%Y-%m-%d')] = coupon / freq
            maturity = maturity - pd.DateOffset(months=int(12/freq))
        tCdf.append(row_data)
    return pd.DataFrame(tCdf).set_index(id_col,drop=True).sort_index(axis=1).fillna(0)

def macDuration(ytm, c, ttm, freq=2):
    ''' MacaulayDuration based on yield-to-maturity [absolute value], coupon [%], 
    time-to-maturity [years] and frequency [periods per year]'''
    tp=0
    tt=((ttm*freq)-int(ttm*freq))/freq
    tpp=0
    if tt == 0:
        tp-=c/freq
    while tt < ttm:
        tp+=(c/freq)*(1/((1+(ytm/freq))**(tt*freq)))
        tpp+=tt*(c/freq)*(1/((1+(ytm/freq))**(tt*freq)))
        tt+=1/freq
    tp+=(100+(c/freq))*(1/((1+(ytm/freq))**(ttm*freq)))
    tpp+=ttm*(100+(c/freq))*(1/((1+(ytm/freq))**(ttm*freq)))
    return tpp/tp

def get_price(yield_, maturity, coupon, freq=2):
    '''Price based on yield [absolute value], maturity [years], coupon [%], 
    and frequency [periods per year]'''
    tp=0
    tt=((maturity*freq)-int(maturity*freq))/freq
    if tt == 0:
        tp-=coupon/freq
    while tt < maturity:
        tp+=(coupon/freq)/(((1+(yield_/freq))**(freq*tt)))
        tt+=1/freq
    tp+=(100+(coupon/freq))/((1+(yield_/freq))**(maturity*freq))
    return tp

def get_yield(price, maturity, coupon, freq=2):
    '''Yield based on price [currency], maturity [years] and coupon [%]'''
    def bond_price(yield_):
        tp=0
        tt=((maturity*freq)-int(maturity*freq))/freq
        if tt == 0:
            tp-=coupon/freq
        while tt < maturity:
            tp+=(coupon/freq)/(((1+(yield_/freq))**(freq*tt)))
            tt+=1/freq
        tp+=(100+(coupon/freq))/((1+(yield_/freq))**(maturity*freq))
        return price-tp
    return opt.root(bond_price, 0.02).x[0]/365.25

def estimate_ns(*args):
    '''Make sure first argument is the C matrix (columns are times and rows are issues) and the second argument is a dataframe with a "price" column and same index.
    Results are called through the .x attribute of the returned object. 0 is t0, 1 is t1, 2 is t2, and 3 is lambda.'''
    def estimate_ns(params, *args):
        M, tba = args[0], args[1]
        t0, t1, t2, lam = params[0], params[1], params[2], params[3]
        tM=M.copy()
        times=tM.columns
        times=(pd.to_datetime(times)-pd.to_datetime('2023-07-23')).days/365.25
        tM.columns=times
        error=0
        for i in tM.index:
            price=0
            tM.loc[i][tM.loc[i]>0].index
            for ii in tM.loc[i][tM.loc[i]>0].index:
                rate=t0+((t1+t2)*((1-np.exp(-ii/lam))/(ii/lam)))-t2*np.exp(-ii/lam)
                price+=tM.loc[i,ii]*np.exp(-rate*ii)
            error+=(price-tba.loc[i,'price'])**2
        return error
    return opt.minimize(estimate_ns, [0.01, 0.01, 0.01, 1], args=(args), method='L-BFGS-B')