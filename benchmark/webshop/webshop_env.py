import re
import os
import pdb
import yaml
import requests
import logging
from bs4 import BeautifulSoup
from bs4.element import Comment
from pathlib import Path
from difflib import get_close_matches
from urllib.parse import quote_plus


def clean_str(p):
    return p.encode('latin-1', errors='ignore').decode('latin-1')


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
            element.parent.name not in ignore and not isinstance(element, Comment)
    )

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}


class Webshop:
    def __init__(self, web_url="http://127.0.0.1:3000"):
        self.sessions = {}
        self.session = None
        self.web_url = web_url
        self.action_space = []
        self.previous_observation = ''
        self.goal = None
        self.reward = 0
        self.sub_reward = 0
        self.obs = None
        self.done = False
        self.history = []
        self.states = []
        self.infos = {}
        self.steps = 0

    def get_info(self):
        pass

    def get_obs(self):
        return self.obs

    def get_goal(self):
        return self.goal

    def get_history(self):
        pass

    def update_item_actions(self, session):
        basic_action_dic = self.basic_action_dic
        # get option_types
        option_types = self.sessions[session].get('option_types', [])
        # generate click[option_type]
        option_actions = [f"click[{option_type}]" for option_type in option_types]
        # update basic_action_dic
        basic_action_dic["item"] = self.basic_action_dic["item"] + option_actions

        return basic_action_dic["item"]

    def update_search_actions(self, session):
        basic_action_dic = self.basic_action_dic
        asins = self.sessions[session].get('asins', [])
        actions = [f"click[{asin}]" for asin in asins]
        basic_action_dic["search"] = self.basic_action_dic["search"] + actions

        return basic_action_dic["search"]

    def get_action_space(self, session):
        self.basic_action_dic = {
            "init": ["search[]"],
            "search": ["click[Next >]", "click[< Prev]", "click[Back to Search]"],
            "item": ["click[Buy Now]", "click[< Prev]", "click[Description]", "click[Features]", "click[Reviews]",
                     "click[Attributes]", "click[Back to Search]"],
            "item_sub": ["click[Back to Search]", "click[< Prev]"]
        }
        assert self.sessions[session]['page_type'] in ['init', 'search', 'item_sub', 'item']
        page_type = self.sessions[session]['page_type']

        if self.sessions[session]['page_type'] == 'item':
            valid_actions = self.update_item_actions(session)
        elif self.sessions[session]['page_type'] == 'search':
            valid_actions = self.update_search_actions(session)
        else:
            valid_actions = self.basic_action_dic[page_type]

        self.action_space = valid_actions
        return valid_actions

    def is_done(self):
        return self.done

    def update(self, action, obs, reward, done):
        self.history.append(("action", action))
        self.history.append(("reward", reward))
        self.history.append(("state", obs))
        self.done = done
        self.states.append(obs)

        self.steps += 1

        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]

    def reset(self):  # Realized in step section
        pass

    def step(self, session, action):
        done = False
        observation_ = None
        grounding = True
        self.session = session
        if action == 'reset[]':
            self.sessions[session] = {'session': session, 'page_type': 'init'}

        elif action.startswith('search['):
            if self.sessions[session]['page_type'] == 'init':
                query = action[7:-1]
                self.sessions[session] = {'session': session, 'page_type': 'search',
                                          'query_string': query, 'page_num': 1}
            else:
                grounding = False
                observation_ = 'There is no [Search] button, you should click the [Back to Search] button first to ' \
                               'search something '
        elif action.startswith('click['):
            button = action[6:-1]
            if button == 'Buy Now':
                if self.sessions[session]['page_type'] == 'item':
                    self.sessions[session]['page_type'] = 'end'
                    done = True
                else:
                    observation_ = "There is no [Buy Now] button"
            elif button == 'Back to Search':
                if self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']:
                    self.sessions[session] = {'session': session, 'page_type': 'init'}
                else:
                    observation_ = "This is no [Back to Search] button"
            elif button == 'Next >':
                if self.sessions[session]['page_type'] == 'search':
                    if self.sessions[session]['page_num'] > 5:
                        pass
                        # raise PageNumberError  # page number limitation
                    self.sessions[session]['page_num'] += 1
                else:
                    observation_ = "This is no [Next >] button"
            elif button == '< Prev':
                if self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']:
                    if self.sessions[session]['page_type'] == 'search':
                        if self.sessions[session]['page_num'] == 1:
                            # raise PageNumberError  # page number limitation
                            pass
                        self.sessions[session]['page_num'] -= 1
                    elif self.sessions[session]['page_type'] == 'item_sub':
                        self.sessions[session]['page_type'] = 'item'
                    elif self.sessions[session]['page_type'] == 'item':
                        self.sessions[session]['page_type'] = 'search'
                        self.sessions[session]['options'] = {}
                else:
                    observation_ = "This is no [< Prev] button"
            elif button in ACTION_TO_TEMPLATE:
                if self.sessions[session]['page_type'] == 'item':
                    self.sessions[session]['page_type'] = 'item_sub'
                    self.sessions[session]['subpage'] = button
                else:
                    observation_ = f"This is no [{button}] button"
            else:
                if self.sessions[session]['page_type'] == 'search':
                    if button in self.sessions[session].get('asins', []):  # must be asins
                        self.sessions[session]['page_type'] = 'item'
                        self.sessions[session]['asin'] = button
                    else:
                        observation_ = f"Button must be an asin"
                elif self.sessions[session]['page_type'] == 'item':
                    if 'option_types' in self.sessions[session]:
                        if button in self.sessions[session]['option_types']:
                            option_type = self.sessions[session]['option_types'][button]
                            if not 'options' in self.sessions[session]:
                                self.sessions[session]['options'] = {}
                            self.sessions[session]['options'][option_type] = button
                            observation_ = f'You have clicked {button}.'
                        else:
                            observation_ = f"your button must be in {self.sessions[session]['option_types']}."
                    else:
                        observation_ = f"No option_types."
        else:
            grounding = False
            observation = 'Incorrect action format. Please use the correct action format following:\n' \
                          'Available Actions:\n\nclick[something]: Engage with specific buttons or links.\n' \
                          'search[something]: Seek specific data on the website. Use this only if a [Search] button ' \
                          'appears in the observation.\n' \
                          'Note: If you wish to search and there is no [Search] button, click the [Back to Search] ' \
                          'button instead. '

            reward = 0.0
            if 'info' in locals():
                reward = info.get('reward', 0.0)
            self.update(action, observation, self.sub_reward, done)
            
            return observation, reward, done, self.sub_reward, grounding

        observation, info = self.webshop_text(**self.sessions[session])
        if observation_:
            observation = observation_
        self.sessions[session].update(info)
        reward = info.get('reward', 0.0)
        pattern = re.compile(r'Instruction:\s*(.*)\s*\[')
        match = pattern.search(observation)
        if match:
            self.goal = match.group(1).strip()
            observation = pattern.sub('', observation).strip()
        observation = "WEB PAGE: {" + observation + "}"
        self.obs = observation
        self.previous_observation = observation  # Recording observation for [think] step

        self.update(action, observation, self.sub_reward, done)
        return observation, reward, done, self.sub_reward, grounding


    def webshop_text(self, session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
        if page_type == 'init':
            url = (
                f'{self.web_url}/{session}'
            )
        if page_type == 'search':
            query_string = quote_plus(query_string)
            url = (
                f'{self.web_url}/search_results/{session}/'
                f'{query_string}/{page_num}'
            )
        elif page_type == 'item':
            query_string = quote_plus(query_string)
            options = {k: quote_plus(v) for k, v in options.items()}
            url = (
                f'{self.web_url}/item_page/{session}/'
                f'{asin}/{query_string}/{page_num}/{options}'
            )
        elif page_type == 'item_sub':
            query_string = quote_plus(query_string)
            options = {k: quote_plus(v) for k, v in options.items()}
            url = (
                f'{self.web_url}/item_sub_page/{session}/'
                f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
            )
        elif page_type == 'end':
            options = {k: quote_plus(v) for k, v in options.items()}
            url = (
                f'{self.web_url}/done/{session}/'
                f'{asin}/{options}'
            )
        # Mark request URL
        request_id = 'Resquest: ' + url
        headers = {'X-Request-ID': request_id}
        html = requests.get(url, headers=headers).text
        html_obj = BeautifulSoup(html, 'html.parser')
        texts = html_obj.findAll(text=True)
        visible_texts = list(filter(tag_visible, texts))
        if False:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            option_type = ''
            options = {}
            asins = []
            cnt = 0
            prod_cnt = 0
            just_prod = 0
            skip_counter = 0
            for i, t in enumerate(visible_texts):
                if skip_counter > 0:
                    skip_counter -= 1
                    continue  # progress score is invisible
                if t == '\n': continue
                if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
                if 'Your progress score (min 0.0, max 1.0)' in t:
                    skip_counter = 1  # skip twice
                    self.reward = float(visible_texts[i + 1])
                    if float(visible_texts[i + 1]) > self.sub_reward:
                        self.sub_reward = float(visible_texts[i + 1])
                    continue  # progress score is invisible
                if t.parent.name == 'button':  # button
                    processed_t = f'\n[{t}] '
                elif t.parent.name == 'label':  # options
                    if f"'{t}'" in url:
                        processed_t = f'[[{t}]]'
                    else:
                        processed_t = f'[{t}]'
                    options[str(t)] = option_type
                elif t.parent.get('class') == ["product-link"]:  # product asins
                    processed_t = f'\n[{t}] '
                    if prod_cnt >= 3:
                        processed_t = ''
                    prod_cnt += 1
                    asins.append(str(t))
                    just_prod = 0
                else:  # regular, unclickable text
                    processed_t = '\n' + str(t) + ' '
                    if cnt < 2 and page_type != 'init': processed_t = ''
                    if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                    option_type = str(t)
                    cnt += 1
                just_prod += 1
                observation += processed_t
            info = {}
            if options:
                info['option_types'] = options
            if asins:
                info['asins'] = asins
            if 'Your score (min 0.0, max 1.0)' in visible_texts:
                idx = visible_texts.index('Your score (min 0.0, max 1.0)')
                info['reward'] = float(visible_texts[idx + 1])
                self.reward = info['reward']
                if float(visible_texts[idx + 1]) > self.sub_reward:
                    self.sub_reward = float(visible_texts[idx + 1])
                observation = 'Result: [Success]' if float(visible_texts[idx + 1]) == 1.0 else 'Result: [False]'
            return clean_str(observation), info
