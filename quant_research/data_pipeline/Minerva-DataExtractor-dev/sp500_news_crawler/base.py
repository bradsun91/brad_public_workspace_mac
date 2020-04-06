import time
from random import randint
from abc import ABCMeta, abstractmethod

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


class RobotBase(object, metaclass=ABCMeta):
    def __init__(self, browser_drive_path):
        options = webdriver.ChromeOptions()
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        self._browser = webdriver.Chrome(
            browser_drive_path, chrome_options=options)

    @abstractmethod
    def run(self):
        pass

    def _sleep(self, min=3, max=5):
        time.sleep(randint(min, max))

    def _maximize_window(self):
        self._browser.maximize_window()

    def _load_url(self, url):
        self._browser.get(url)

    def _get_bs(self):
        return BeautifulSoup(self._browser.page_source, "lxml")

    def _find_element_by_id(self, id):
        try:
            element = self._browser.find_element_by_id(id)
            return element
        except:
            return None

    def _find_element_by_css_selector(self, css_selector):
        try:
            element = self._browser.find_element_by_css_selector(css_selector)
            return element
        except:
            return None

    def _find_elements_by_css_selector(self, css_selector):
        try:
            elements = self._browser.find_elements_by_css_selector(
                css_selector)
            return elements
        except:
            return None

    def _scroll_to_bottom(self):
        self._browser.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        self._sleep()

    def _scroll_to_bottom_by_css_selector(self, css_selector):
        element = self._find_element_by_css_selector(css_selector)
        self._browser.execute_script(
            "arguments[0].scrollTo(0, arguments[0].scrollHeight);", element)
        self._sleep()

    def _click(self, element):
        ac = ActionChains(self._browser).move_to_element(element)
        ac.click().perform()
        self._sleep()

    def _click_by_id(self, id, is_scroll_to_bottom=False):
        element = self._find_element_by_id(id)
        if is_scroll_to_bottom:
            self._scroll_to_bottom()
        ac = ActionChains(self._browser).move_to_element(element)
        ac.click().perform()
        self._sleep()

    def _click_by_css_selector(self, css_selector, is_scroll_to_bottom=False):
        element = self._find_element_by_css_selector(css_selector)
        if is_scroll_to_bottom:
            self._scroll_to_bottom()
        ac = ActionChains(self._browser).move_to_element(element)
        ac.click().perform()
        self._sleep()

    def _fillin_input_element_by_id(self, id, value):
        input = self._find_element_by_id(id)
        input.send_keys(value)
        self._sleep()

    def _fillin_input_element_by_css_selector(self, css_selector, value):
        input = self._find_element_by_css_selector(css_selector)
        input.send_keys(value)
        self._sleep()

    def _switch_to_frame(self, iframe):
        self._browser.switch_to.frame(iframe)

    def _switch_to_default(self):
        self._browser.switch_to.default_content()

    def _move_to_element_by_css_selector(self, css_selector):
        element = self._find_element_by_css_selector(css_selector)
        ac = ActionChains(self._browser).move_to_element(element)
        ac.perform()
