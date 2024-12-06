class Person:
    '''python中__call__的用法'''
    def __call__(self, name):
        print('__call__' + 'hello' + name)

    def hello(self, name):
        print('hello' + name)

person = Person()
person('zhangsan') # 调用__call__
person.hello('lisi')
