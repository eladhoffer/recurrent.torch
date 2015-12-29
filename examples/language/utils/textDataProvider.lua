function loadTextFileWords(filename, vocab)

    local file = io.open(filename, 'r')
    local words = file:read("*all"):split(' ')
    local length = #words
    local wordsVec = torch.LongTensor(#words):zero()

    local vocab = vocab or {['\n'] = 1}
    local currentNum = 1
    --count num words (in case of existing vocab)
    for _ in pairs(vocab) do currentNum = currentNum + 1 end

    for i=1, length do
        local currWord = words[i]
        local encodedNum = vocab[currWord]
        if not encodedNum then
            vocab[currWord] = currentNum
            encodedNum = currentNum
            currentNum = currentNum + 1
        end
        wordsVec[i] = encodedNum
    end

    local decoder = {}
    for word, num in pairs(vocab) do
        decoder[num] = word
    end
    return wordsVec, vocab, decoder
end

function loadTextFileChars(filename, vocab)
  local file = torch.DiskFile(filename, 'r')
  file:seekEnd()
  local length = file:position() - 1
  file:seek(1)
  local byteVec = torch.ByteTensor(length)
  file:readByte(byteVec:storage())

  local vocab = vocab or {}
  local currentNum = 1
  local data = byteVec:data()
  for i=0, length-1 do
    local encodedNum = vocab[data[i]]
    if not encodedNum then
      vocab[data[i]] = currentNum
      encodedNum = currentNum
      currentNum = currentNum + 1
    end
    data[i] = encodedNum
  end
  local decoder = {}
  for val, num in pairs(vocab) do
    decoder[num] = string.char(val)
  end
  return byteVec, vocab, decoder
end

function decodeFunc(decoder, mode)
  local space = ''
  if mode == 'word' then
    space = ' '
  end
  local func = function(vec)
    local output = ''
    for i=1, vec:size(1) do
        output = output .. space .. decoder[vec[i]]
    end
    return output
  end
  return func
end


 function encodeFunc(vocab, mode)
  local func
  if mode == 'word' then
    func = function(str)
      local words = str:split(' ')
      local length = #words
      local encoded = torch.LongTensor(#words):zero()

      for i=1, length do
          local currWord = words[i]
          local encodedNum = vocab[currWord]
          if not encodedNum then
              encodedNum = -1
          end
          encoded[i] = encodedNum
      end
      return encoded
    end
  elseif mode == 'char' then
    func = function(str)
    local length = #str
    local encoded = torch.ByteTensor(length):zero()

    for i=1, length do
      local encodedNum = vocab[str[i]]
      if not encodedNum then
          encodedNum = -1
        end
        encoded[i] = encodedNum
      end
      return encoded
    end
  end
    return func
end
