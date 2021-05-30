import os
import pickle
import time
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup


main_url = "http://sjw.history.go.kr/main.do"
main_page = BeautifulSoup(requests.get(main_url).text, 'lxml')
print(main_url)

main_dir = "/home/nas_datasets/joseon_record/crawledResults_sjw"
if not os.path.exists(main_dir):
    os.mkdir(main_dir)

king_areas_1 = main_page.find("div", {"class": "m_cont_top"}).findAll("li")


def main():
    articles = list()
    for king_area_1 in king_areas_1[1:9]:
        area_text = king_area_1.text
        print(area_text)
        area_dir = f'{main_dir}/{area_text}'

        if not os.path.exists(area_dir):
            os.mkdir(area_dir)

        area_id = king_area_1.find('a')['href'][-14:-9]
        url = f"http://sjw.history.go.kr/search/inspectionMonthList.do?TreeID={area_id}"
        try:
            area_page = BeautifulSoup(requests.post(url).text, 'lxml')
        except:
            time.sleep(100)
            area_page = BeautifulSoup(requests.post(url).text, 'lxml')
            print(str(king_area_1) + "|||")

        for year in area_page.find('ul', {'class': 'king_year2 clear2'}).findAll('ul'):
            for month in year.findAll('a'):
                articles = []
                id_ = month['href'].split("'")[1]
                url = f"http://sjw.history.go.kr/search/inspectionDayList.do?TreeID={id_}"
                try:
                    month_page = BeautifulSoup(requests.get(url).text, 'lxml')
                except:
                    time.sleep(100)
                    month_page = BeautifulSoup(requests.get(url).text, 'lxml')

                book_month_page = month_page.find("div", {'class': 'view_tit'})
                book_section_text = book_month_page.find('span').text.strip()
                book_section_text = book_section_text.split("월")[0] + "월"
                print(book_section_text)

                with open(f"{area_dir}/{book_section_text}.pkl", 'wb') as fwrite:
                    month_page = month_page.find('span', {'class': 'day_list'})
                    for event in month_page.findAll('a'):
                        if event['href'] == '#':
                            event_id = quote_plus(id_)
                        else:
                            event_id = event['href'].split("'")[1]
                            event_id = quote_plus(event_id)

                        event_url = f"http://sjw.history.go.kr/search/inspectionDayList.do?treeID={event_id}"
                        try:
                            event_page = BeautifulSoup(requests.get(event_url).text, 'lxml')
                        except:
                            time.sleep(100)
                            event_page = BeautifulSoup(requests.get(event_url).text, 'lxml')

                        time.sleep(0.2)

                        book_month_page = event_page.find("div", {'class': 'view_tit'})
                        book_section_text = book_month_page.find('span').text.strip()
                        event_time = book_section_text.split("\r")[0] + book_section_text.split("\t")[4]
                        date = " ".join(event_time.split()[3:7])

                        try:
                            name_text = event_page.find('h2', {'class': 'search_tit'}).text
                        except AttributeError: # Because of NoneType
                            continue

                        if name_text == "":
                            name_text = "\t"

                        event_chinese = event_page.find('ul', {'class': 'sjw_list'})
                        for paragraph_chinese in event_chinese.findAll('a'):
                            try:
                                chinese_text = paragraph_chinese.find('span').text
                            except AttributeError: # Because of NoneType
                                continue

                            for chinese_char in chinese_text.split('○'):
                                hanja = chinese_char
                                if not hanja == "":
                                    hanja = " ".join(chinese_char.split())
                                    # Some special case start with '〈○〉'
                                    if not hanja == '〈':
                                        if hanja[:2] == '〉 ':
                                            hanja = hanja[2:]
                                        articles.append({'hanja': hanja, 'date': date})

                    pickle.dump(articles, fwrite)

                time.sleep(3)

if __name__=='__main__':
    main()
