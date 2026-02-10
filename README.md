# Hands-on learning LLM internals with Claude

This is a human learning project, performed with the use of AI tools.

I learn best through hands-on building and experimentation. In this project, I worked with Claude Code only as a tutor. The Rust in `src/` is largely written by my hands, standing on the shoulders of giants through the use of PyTorch. The role of Claude was to teach me what I didn't know (or know well) about LLMs and modern NLP machine learning.

Claude provided an architecture and implementation plan for LLM training and inference, which I then implemented. Throughout the process, Claude would answer my questions (Socratically), and write unit tests after I was done to confirm that the code worked as intended. 

Occasionally, I asked Claude to stub out function signatures and docs for me when we got into really new domains. I would ask for explanations of the concepts, and avoided accepting too many concepts blindly (though many I did -- I'm certainly not an ML engineer after this effort alone).

It was fun and very convenient to have an always-on, ready-when-you-are, custom tutor. I think this is the future of learning for certain people who, like me, learn well by building labs. Vibe-learning, if you will.

See [PLAN.md](./PLAN.md) for details about what I built.

# Why?

> "You don't have to be an engineer to be be a racing driver, but you do have to have Mechanical Sympathy." 
  - *Jackie Stewart, racing driver*

[Mechanical sympathy](https://en.wiktionary.org/wiki/mechanical_sympathy) is the idea that one is more effective with a tool when they understand how the tool works. The originating quote had to do with the automobile, in the context of competitive driving. But the same concept extends to any tool, from aircraft to circular saws. I believe AI, specifically LLMs and their descendents, are the Jacquard loom of knowledge work. Understanding its internal workings yields insights that are valuable in application of the tool.

# Does it work?

As of this commit, and with very low expectations, yes it does. This model uses characters as tokens, to keep things simple. I trained it on 1000 files in the Python standard library, over 5 epochs. The average loss by the end of the final epoch was around 0.5. This was enough to produce Pythonic gibberish. For example, prompting it to complete this function:

``` python
def squares(n): return [x**2 for x in range(n)]
```

Gets a run-on response, but which does correctly begin with the `range` function:

```
? def squares(n): return [x**2 for x in 
range(3)].join(5)
        if x == 0: range(1,'line'): return x*2
        self.assertEqual(self.unc, self.failname)
        self.assertEqual(self.unc, x*2 -0)
        self.assertEqual(self.unc, self.loads(decoded))
        self.assertEqual(self.unc, self.loads(decoded))
        self.assertEqual(self.unc, self.loads(decoded))
        self.assertRaises(StopIteration, self.fail, decoded)
        self.assertRaises(StopIteration, self.decoded, [1, 2, 2, 3, 4, 4, 4,
                2, 2, *,3,
           NoNaNAN,
            -1, 5, -6, -6,
            -1, 6, -13,
            -1, 5, | -13,
           -1, 6, -14, 15, -16, 1,
            #-1,52 :9, 16,
          #-1.118,16,'70>20',1-16,16,17,p0,26,T-1,+19-11,0:12.12
            -1.19  EARM
            (-1,2), -1)

        self.assertRaises(StopIteration, self.esc)
        self.assertRaises(ZeroDivisionError, self.decoded, [-1,2,1)]

        self.assertRaises(StopIteration, self.decoded, [1,2, 3, (2,3, (2,6)])

        self.assertRaises(ZeroDivisionError, self.decoded)
        self.assertRaises(PermissionError, self.decoded)

        del 0
        self.assertRaises(ZeroDivisionError, decoded)

        self.assertRaises(KeyError, decoded, [1,2,3,[4,5,6,6,7,9,110,110,20,2,20,4,6,4,a10,10,20,0,2.0,"a1",20,2~A)
        self.assertRaises(KeyError, decoded, [2,2,2,3,[3,5,6,6,6,10,2,3,2,2]]])
        self.assertRaises(KeyError, decoded, [[b,+10,2,-8,20,-1,4,20]],
             b'abc=%d' , decoded, [b, 3,.2], b], decoded, decoded, (x,2), decoded)
        self.assertRaises(ZeroDivisionError, decoded, [],])

       self.assertEqual(decoded, od, {1:2:2, * 1:4, 2:3, 9}, decoded,
                (1:2, 1, 1, *[], 6, 1, 0, 2, -1], decoded, 11), decoded),
            (1:2, 1, (2:3, 1, -1, False, 1:3, False),
            (3, 1, 6, 1, 2), decoded, index),
          (1:2, 1, 12),
          ([(2, 2, :2, 3, "false", 5, EN", No=[]], decoded, decoded]),
          [(1, 2, 3, "false", 13, ", 2),
        ]:
        end = end.end
        self.assertEqual(_end, {"false": "nan"})  # httest beginnned int
?
```

# Did you pass the test?

I have not taken the test yet. I'll ask my professor when it will be. ;)