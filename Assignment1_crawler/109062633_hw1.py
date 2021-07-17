import requests
from bs4 import BeautifulSoup

def get_soup(target):
    url = 'https://www.blockchain.com/eth/address/' + target.strip() + '?view=standard'
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'lxml')
    return soup

def info_transaction(soup):
    info = soup.find('div', {'class':'hnfgic-0 enzKJw'})
    all_info = [i.text for i in info.find_all('div', {'class':'sc-8sty72-0 bFeqhe'})][2:]
    info_dict = {all_info[i]: all_info[i+1] for i in range(0,12,2)}
    trans_div = soup.find('div', class_='sc-1d6wz2a-0 ewdTvA')
    date = [trans.text for trans in trans_div.find_all('div', {'class':'sc-1rk8jst-0 eFowch'})][::-1]
    account = [trans.text for trans in trans_div.find_all('div',{'class':'sc-1rk8jst-2 flPSdq'})][::-1]
    amount = [trans.text for trans in trans_div.find_all('div', {'class':'sc-1rk8jst-1 bFopoh'})][::-1]
    last_trans_out = [trans for trans in amount if trans.startswith('-')]
    if last_trans_out:
        trans_amount = last_trans_out[0]
        index = amount.index(trans_amount)
        info_dict['Date'] = date[index]
        info_dict['To'] = account[index]
        info_dict['Amount'] = trans_amount
    return info_dict

def write_file(fin_info, address):
    with open('109062633_hw1_output.txt', 'a') as g:
        for info in fin_info:
            address_info = f"Nonce: {info['Nonce']}\nNumber of Transactions: {info['Number of Transactions']}\nFinal Balance: {info['Final Balance']}\nTotal Sent: {info['Total Sent']}\nTotal Received: {info['Total Received']}\nTotal Fees: {info['Total Fees']}\n"

            if 'To' in info:
                transfer_info = f"Date: {info['Date']}\nTo: {info['To']}\nAmount: {info['Amount']}\n"
            else:
                transfer_info = ''

            sep = '--------------------------------------------------------------------------\n'
            g.write(address_info+transfer_info+sep)
        g.write(' -> '.join(address)+'\n'+sep)
        
def main():
    with open('ex_input_hw1.txt', 'r') as f:
        for line in f.readlines():
            addresses, final_info = [], []
            target = line
            for i in range(4):
                addresses.append(target.strip())
                address_soup = get_soup(target)
                info = info_transaction(address_soup)
                final_info.append(info)
                if 'To' in info:
                    target = info['To']
                else:
                    break
            write_file(final_info, addresses)

if __name__ == '__main__':
    main()
