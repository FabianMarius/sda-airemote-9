import unittest

class TestStringOperations(unittest.TestCase):
    def test_string_upper(self):
        self.assertEqual("hello".upper(), "HELLO")

    def test_string_split(self):
        self.assertEqual("a,b,c".split(","), ["a", "b", "c"])


class TestListBehavior(unittest.TestCase):
    def test_list_length(self):
        self.assertEqual(len([1, 2, 3]), 3)


# class TestIntentionalFailure(unittest.TestCase):
#     def test_wrong_sum(self):
#         # This test is intentionally wrong to show what a failure looks like
#         self.assertEqual(1 + 1, 3)



class TestBasicArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass executed once before all tests.")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass executed once after all tests.")

    def setUp(self):
        print("Setup code executed before each test")

    def tearDown(self):
        print("Teardown code executed after each test")

    def test_addition(self):
        self.assertEqual(2 + 3, 5)

    def test_subtraction(self):
        self.assertEqual(10 - 4, 6, "Mesaj in cazul in care testul pica")

    def test_raise_zero_division_error(self):
        with self.assertRaises(ZeroDivisionError):
            result  = 20 / 0 # testul trece, se arunca eroarea

            # testul pica, eroarea nu se arunca
            # result = 20 / 2
            # self.assertEqual(result, 10)


if __name__ == "__main__":
    # condul acesta va rula doar daca programul porneste din acest script
    unittest.main()

